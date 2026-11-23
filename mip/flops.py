"""Estimate FLOPs of executed eager tensor operations without replaying a model.

Matrix/convolution formulas follow ``torch.utils.flop_counter`` (PyTorch 2.2+
through 2.13): two FLOPs per multiply-add. Unlike its counting mode, this observer
does not request decompositions or execute additional control-flow branches.
Unsupported arithmetic is reported, so ``counted_flops`` is a partial analytic
estimate, never a claim about all hardware instructions or elapsed compute.

Fused RNN affine work follows the parameter shapes documented in
https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html and the GRU page.
Fused RNN backward uses two times its forward affine work as an estimate;
backend-internal gradient masks, gate nonlinearities and dropout are omitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from inspect import signature
from math import prod
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

# These operations move/allocate/select data or manage metadata/RNG. Their
# internal instruction counts are deliberately outside tensor-math FLOPs.
_EXCLUDED_NAMES = {
    "alias",
    "arange",
    "as_strided",
    "as_strided_",
    "bernoulli",
    "bernoulli_",
    "cat",
    "clone",
    "contiguous",
    "copy_",
    "detach",
    "detach_",
    "dim",
    "empty",
    "empty_like",
    "empty_strided",
    "expand",
    "expand_as",
    "fill_",
    "full",
    "full_like",
    "gather",
    "index",
    "index_select",
    "is_contiguous",
    "is_non_overlapping_and_dense",
    "is_strides_like_format",
    "lift_fresh",
    "lift_fresh_copy",
    "narrow",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "normal",
    "normal_",
    "numel",
    "ones",
    "ones_like",
    "permute",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "randperm",
    "reshape",
    "repeat",
    "resize_",
    "scalar_tensor",
    "select",
    "set_",
    "size",
    "slice",
    "split",
    "split_with_sizes",
    "squeeze",
    "stack",
    "storage_offset",
    "stride",
    "sym_numel",
    "sym_size",
    "sym_storage_offset",
    "sym_stride",
    "t",
    "to",
    "transpose",
    "unbind",
    "uniform_",
    "unsqueeze",
    "unsqueeze_",
    "view",
    "view_as",
    "zero_",
    "zeros",
    "zeros_like",
    "_local_scalar_dense",
    "_to_copy",
    "_unsafe_view",
}
_EXCLUDED_OPS = {
    "profiler._record_function_enter",
    "profiler._record_function_enter_new",
    "profiler._record_function_exit",
}
_FUSED_RNN_NAMES = {
    "_cudnn_rnn",
    "_cudnn_rnn_backward",
    "miopen_rnn",
    "miopen_rnn_backward",
    "mkldnn_rnn_layer",
    "mkldnn_rnn_layer_backward",
}
_ELEMENTWISE_NAMES = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "neg",
    "addcdiv",
    "addcmul",
    "lerp",
}


def _bind_arguments(func, args, kwargs) -> dict[str, Any]:
    """Use schemas rather than backend/version-specific positional offsets."""
    values = dict(kwargs)
    for argument, value in zip(func._schema.arguments, args, strict=False):
        values[argument.name] = value
    for argument in func._schema.arguments:
        if argument.name not in values and argument.has_default_value():
            values[argument.name] = argument.default_value
    return values


def _elementwise_flops(name: str, values: dict[str, Any], output) -> int | None:
    """Count scalar expressions once per real floating output element.

    Add/sub/mul/div/pow/sqrt/reciprocal/neg cost one scalar operation; rsqrt
    costs two. This assigns powers and roots a mathematical operation, not a
    hardware instruction cost. Lerp is subtract/multiply/add (three).
    Addcdiv/addcmul cost two plus a multiply for non-unit scaling; add/sub
    similarly include non-unit alpha. Tensor-valued scales are never read or
    synchronized: count their multiply even if their runtime value is one.
    """
    base = name.removeprefix("_foreach_").removesuffix("_")
    if base not in _ELEMENTWISE_NAMES:
        return None
    # In-place foreach and foreach out overloads return no tensor.
    if output is None or (isinstance(output, (tuple, list)) and not output):
        output = values.get("out", values.get("self"))
    tensors = list(output) if isinstance(output, (tuple, list)) else [output]
    if any(
        isinstance(tensor, torch.Tensor) and tensor.is_complex() for tensor in tensors
    ):
        return None
    if not any(
        isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        for tensor in tensors
    ):
        return None
    scale = values.get("alpha", 1)
    if base in {"addcdiv", "addcmul"}:
        scale = values.get("scalars", values.get("value", 1))
    scales = scale if isinstance(scale, (list, tuple)) else [scale] * len(tensors)
    total = 0
    for tensor, factor in zip(tensors, scales, strict=True):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        cost = 3 if base == "lerp" else 2 if base == "rsqrt" else 1
        if base in {"addcdiv", "addcmul"}:
            cost = 2
        if base in {"add", "sub", "addcdiv", "addcmul"}:
            cost += int(isinstance(factor, torch.Tensor) or factor != 1)
        total += int(tensor.numel()) * cost
    return total


def _fused_rnn_flops(name: str, values: dict[str, Any]) -> int:
    """Count all layer/direction/projection affine weights once per time point."""
    inputs = values["input"]
    positions = prod(inputs.shape[:-1])
    if name.startswith("mkldnn"):
        # oneDNN exposes one layer/direction per operator, even for stacked RNNs.
        keys = ("weight1", "weight2") if "backward" in name else ("weight0", "weight1")
        weights = [values[key] for key in keys]
    else:
        # cuDNN/MIOpen expose all layers/directions in this list. Bias vectors
        # are excluded; projected LSTM matrices are included by their rank.
        weights = values["weight"]
    affine_weights = sum(int(weight.numel()) for weight in weights if weight.ndim == 2)
    if not affine_weights:
        raise ValueError("Fused RNN exposes no two-dimensional affine weights")
    flops = 2 * int(positions) * affine_weights
    return flops * (2 if "backward" in name else 1)


def _attention_matmul_flops(query, key, value) -> int:
    # Dense attention: Q @ K.T followed by probability @ V. This deliberately
    # follows PyTorch's dense convention, without discounts for masks/causality.
    batch_heads = prod(query.shape[:-2])
    query_len, key_dim = query.shape[-2:]
    key_len = key.shape[-2]
    value_dim = value.shape[-1]
    return int(2 * batch_heads * query_len * key_len * (key_dim + value_dim))


def _native_attention_flops(values: dict[str, Any], *, transformer=False) -> int:
    """Account for fused inference attention projections and optional FFN."""
    query = values["src"] if transformer else values["query"]
    if query.is_nested or query.ndim != 3:
        raise ValueError(
            "Fused attention requires a dense batch/sequence/features input"
        )
    batch, length, embed = query.shape
    if not transformer and (
        values["key"].shape != query.shape or values["value"].shape != query.shape
    ):
        raise ValueError("Fused native cross-attention shape is not covered")
    weights = [values["qkv_weight"], values["proj_weight"]]
    if transformer:
        weights.extend([values["ffn_weight_1"], values["ffn_weight_2"]])
    projections = 2 * batch * length * sum(weight.numel() for weight in weights)
    attention = 4 * batch * length * length * embed
    return int(projections + attention)


class ExecutedFlopCounter(TorchDispatchMode):
    """Observe one eager execution, including autograd and checkpoint recompute.

    Example::

        with ExecutedFlopCounter() as counter:
            loss = model(inputs).square().mean()
            loss.backward()
        report = counter.snapshot()

    No hooks, extra forwards, parameter changes, RNG resets, or model-mode
    changes are performed. A fresh context resets the counters. Backward is
    identified by the autograd engine; other work defaults to ``forward``.
    Use ``counter.phase("optimizer")`` to name additional phases explicitly.
    Higher-order/custom/compiled operators may be opaque and are reported as
    unsupported when visible; kernels bypassing Python dispatch are not seen.
    """

    supports_higher_order_operators = True

    def __init__(self):
        super().__init__()
        self._registry = dict(flop_registry)
        self._registry_output_keywords = {}
        self._active = False
        self._phase_override = None
        self._reset()

    def _reset(self):
        self.operator_flops = Counter()
        self.operator_calls = Counter()
        self.phase_flops = Counter()
        self.unsupported_compute_ops = Counter()
        self.ignored_ops = Counter()
        self.partially_counted_ops = {}
        self.formula_failures = {}
        self._phase_operator_flops = defaultdict(Counter)
        self.completed = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("ExecutedFlopCounter cannot reenter the same instance")
        self._reset()
        super().__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._active = False
            self.completed = exc_type is None

    @contextmanager
    def phase(self, name: str):
        """Explicitly label work whose phase cannot be inferred from autograd."""
        previous = self._phase_override
        self._phase_override = str(name)
        try:
            yield self
        finally:
            self._phase_override = previous

    @property
    def counted_flops(self) -> int:
        return int(sum(self.operator_flops.values()))

    def _estimate(self, func, args, kwargs, output):
        packet = getattr(func, "_overloadpacket", None)
        name = str(packet).removeprefix("aten.")
        if name.removeprefix("_foreach_").removesuffix("_") in _ELEMENTWISE_NAMES:
            return _elementwise_flops(
                name, _bind_arguments(func, args, kwargs), output
            ), (
                "Real floating scalar arithmetic convention; powers/roots count as scalar ops, "
                "non-unit scaling adds a multiply, tensor scale values are not inspected."
            )
        if name in _FUSED_RNN_NAMES:
            values = _bind_arguments(func, args, kwargs)
            return _fused_rnn_flops(name, values), (
                "Affine work only; gate/bias/dropout arithmetic omitted. Fused backward "
                "is estimated as twice forward affine work, without backend gradient masks."
            )
        if name in {
            "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_flash_attention_for_cpu_backward",
        }:
            values = _bind_arguments(func, args, kwargs)
            count = _attention_matmul_flops(
                values["query"], values["key"], values["value"]
            )
            return count * (2 if "backward" in name else 1), (
                "Dense attention matmuls only; softmax, scaling, masking and dropout omitted."
            )
        if name in {"_native_multi_head_attention", "_transformer_encoder_layer_fwd"}:
            return (
                _native_attention_flops(
                    _bind_arguments(func, args, kwargs),
                    transformer=name == "_transformer_encoder_layer_fwd",
                ),
                "Dense projections/attention/FFN only; bias, normalization and activations omitted.",
            )
        if packet in self._registry:
            formula = self._registry[packet]
            if formula not in self._registry_output_keywords:
                parameters = signature(formula, follow_wrapped=False).parameters
                # PyTorch 2.2 wrappers use out=; newer versions use out_val=.
                self._registry_output_keywords[formula] = (
                    "out"
                    if "out" in parameters and "out_val" not in parameters
                    else "out_val"
                )
            output_keyword = self._registry_output_keywords[formula]
            formula_kwargs = {**kwargs, output_keyword: output}
            return int(formula(*args, **formula_kwargs)), (
                "PyTorch dense tensor-math formula; fused bias/nonlinear/mask arithmetic may be omitted."
            )
        return None, None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Execute exactly the original operator once. In particular, never call
        # func.decompose(): doing so can change kernels, RNG or control flow.
        output = func(*args, **kwargs)
        opname = str(func)
        self.operator_calls[opname] += 1
        phase = self._phase_override
        if phase is None:
            phase = "backward" if torch._C._current_graph_task_id() != -1 else "forward"
        try:
            count, omitted = self._estimate(func, args, kwargs, output)
        except Exception as exc:
            # A profiler formula must not turn a successful training op into a
            # failed update. Preserve the gap explicitly instead of adding zero.
            self.formula_failures[opname] = f"{type(exc).__name__}: {exc}"
            count, omitted = None, None
        if count is not None:
            self.operator_flops[opname] += count
            self.phase_flops[phase] += count
            self._phase_operator_flops[phase][opname] += count
            self.partially_counted_ops[opname] = omitted
        else:
            packet = str(getattr(func, "_overloadpacket", ""))
            if packet in _EXCLUDED_OPS or (
                packet.startswith("aten.") and packet[5:] in _EXCLUDED_NAMES
            ):
                self.ignored_ops[opname] += 1
            else:
                self.unsupported_compute_ops[opname] += 1
        return output

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe counts and the omissions needed to interpret them."""
        return {
            "schema_version": 1,
            "backend": "torch_dispatch_eager",
            "torch_version": str(torch.__version__),
            "flops_per_mac": 2,
            "counted_flops": self.counted_flops,
            "forward_flops": int(self.phase_flops["forward"]),
            "backward_flops": int(self.phase_flops["backward"]),
            "phase_flops": dict(self.phase_flops),
            "operator_flops": dict(self.operator_flops),
            "phase_operator_flops": {
                phase: dict(counts)
                for phase, counts in self._phase_operator_flops.items()
            },
            "operator_calls": dict(self.operator_calls),
            "unsupported_compute_ops": dict(self.unsupported_compute_ops),
            "ignored_ops": dict(self.ignored_ops),
            "ignored_noncompute_ops": dict(self.ignored_ops),
            "partially_counted_ops": dict(self.partially_counted_ops),
            "formula_failures": dict(self.formula_failures),
            "coverage": "partial_estimate"
            if self.counted_flops
            else "no_supported_math_counted",
            "completed": self.completed,
            "limitations": [
                "Counts observed eager dense tensor math, not all FLOPs or hardware instructions.",
                "Missing arithmetic appears in unsupported_compute_ops; counted fused ops list omissions.",
                "Shape, allocation, memory/indexing, RNG and profiler markers in ignored_ops are excluded.",
                "Non-backward work is labelled forward unless explicitly assigned a phase.",
                "Backward includes checkpoint recomputation executed by the autograd engine.",
                "Compiled/custom kernels bypassing Python dispatch cannot be measured by this observer.",
            ],
        }
