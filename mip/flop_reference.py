"""Isolated eager reference counts for compiled/CUDA-graph policy calls."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
import torch

from mip.flop_accounting import FlopLedger
from mip.flops import ExecutedFlopCounter


@contextmanager
def _preserve_rng():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    numpy = np.random.get_state()
    python = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy)
        random.setstate(python)


def _cpu_memo(value, memo, seen):
    """Copy tensors directly to CPU, including aliases and non-leaf caches.

    Building the memo before deepcopy avoids first duplicating GPU models and
    optimizer state on the GPU. No live parameter or buffer is modified.
    """
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", copy=True)
        if isinstance(value, torch.nn.Parameter):
            tensor = torch.nn.Parameter(tensor, requires_grad=value.requires_grad)
            if value.grad is not None:
                tensor.grad = value.grad.detach().to(device="cpu", copy=True)
        elif value.requires_grad:
            tensor.requires_grad_(True)
        memo[identity] = tensor
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _cpu_memo(key, memo, seen)
            _cpu_memo(item, memo, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _cpu_memo(item, memo, seen)
    elif isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
        _cpu_memo(vars(value), memo, seen)


def _make_reference(agent, args):
    shadow = copy.copy(agent)
    shadow.config = copy.deepcopy(agent.config)
    shadow.config.optimization.device = "cpu"
    memo, seen = {}, set()
    copied = {
        name: value
        for name, value in vars(agent).items()
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer))
    }
    _cpu_memo(copied, memo, seen)
    _cpu_memo(args, memo, seen)
    for name, value in copied.items():
        setattr(shadow, name, copy.deepcopy(value, memo))
    inputs = copy.deepcopy(args, memo)
    shadow.use_compile = False
    # Retain the existing CUDA-graph branch's mode/EMA semantics, but execute
    # its mathematical body eagerly on the isolated CPU modules.
    shadow._compiled_update = shadow._create_update_impl()
    shadow._compiled_sampler = (
        shadow._inference_mode()(shadow._sample_impl)
        if shadow.use_cudagraphs
        else shadow._sample_impl
    )
    shadow.flops = FlopLedger(shadow.config, enabled=False)
    return shadow, inputs


def profile_reference(agent, phase, args):
    """Count a same-shape CPU reference without changing live compiled execution.

    Compiler fusion and hardware-specific kernels can have different instruction
    counts. The report labels this reference as an arithmetic estimate. CPU
    memory must accommodate the representative batch and model during profiling.
    """
    with _preserve_rng():
        shadow, inputs = _make_reference(agent, args)
        with torch.enable_grad(), ExecutedFlopCounter() as counter:
            if phase == "train":
                shadow._update_untracked(*inputs)
            else:
                shadow._sample_untracked(*inputs)
        report = counter.snapshot()
    report["reference"] = {
        "device": "cpu",
        "batch_size": int(args[0].shape[0]),
        "method": "isolated same-shape eager reference for compiled execution",
        "compiled_kernel_instructions_measured": False,
    }
    return report
