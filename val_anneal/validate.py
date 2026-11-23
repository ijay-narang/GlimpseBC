"""Final, offline validation for the isolated annealing experiments.

This module deliberately knows nothing about simulator environments.  It runs the
existing policy sampler over every held-out sequence and aggregates errors by
element, so a short final DataLoader batch is weighted correctly.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mip.losses import get_norm


def _setting(config: Any, name: str, default: Any) -> Any:
    validation = getattr(config, "validation", None)
    return getattr(validation, name, default) if validation is not None else default


def _prepare_observations(config: Any, raw_obs: Any) -> Any:
    """Mirror the three benchmark training loops without importing an env."""
    device = config.optimization.device
    obs_steps = int(config.task.obs_steps)
    if isinstance(raw_obs, Mapping) or hasattr(raw_obs, "items"):
        return {
            key: value[:, :obs_steps].to(device, non_blocking=True)
            for key, value in raw_obs.items()
        }
    return raw_obs[:, :obs_steps].to(device, non_blocking=True)


@contextmanager
def _isolated_rng(seed: int):
    """Make sampling repeatable without consuming the caller's RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def _module_modes(agent: Any) -> list[tuple[torch.nn.Module, bool]]:
    result: list[tuple[torch.nn.Module, bool]] = []
    seen: set[int] = set()
    for name in ("encoder", "flow_map", "encoder_ema", "flow_map_ema"):
        module = getattr(agent, name, None)
        if isinstance(module, torch.nn.Module) and id(module) not in seen:
            seen.add(id(module))
            result.append((module, bool(module.training)))
    return result


def _anneal_steps(agent: Any) -> dict[str, torch.Tensor]:
    """Snapshot only curriculum buffers; validation must leave them unchanged."""
    result: dict[str, torch.Tensor] = {}
    for module_name in ("encoder", "encoder_ema"):
        module = getattr(agent, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        for name, value in module.named_buffers():
            if name.rsplit(".", 1)[-1] == "anneal_step":
                result[f"{module_name}.{name}"] = value.detach().cpu().clone()
    return result


def _restore_modes(modes: list[tuple[torch.nn.Module, bool]]) -> None:
    for module, was_training in modes:
        module.train(was_training)


def _unnormalize(action_normalizer: Any, value: torch.Tensor) -> np.ndarray:
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(action_normalizer.unnormalize(array), dtype=np.float64)


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_offline(
    config: Any,
    dataset: Any,
    agent: Any,
    *,
    normalizer: Any | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    num_steps: int | None = None,
) -> dict[str, float | int]:
    """Evaluate EMA predictions on every held-out sequence exactly once.

    ``agent.eval()`` is essential here: annealed vanilla then takes its final
    hard crop, while annealed foveated uses its final fixed radius (and retains
    Gaussian foveation).  Neither encoder advances ``anneal_step`` in eval mode.
    """
    if len(dataset) <= 0:
        raise ValueError("offline validation dataset is empty")

    action_normalizer = (
        normalizer if normalizer is not None else getattr(dataset, "normalizer", None)
    )
    if isinstance(action_normalizer, Mapping):
        action_normalizer = action_normalizer.get("action")
    if action_normalizer is None or not hasattr(action_normalizer, "unnormalize"):
        raise TypeError(
            "a training-set action normalizer with unnormalize() is required"
        )

    batch_size = int(
        batch_size
        if batch_size is not None
        else _setting(config, "batch_size", config.optimization.batch_size)
    )
    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    if num_workers is None:
        num_workers = int(_setting(config, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("validation num_workers cannot be negative")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=str(config.optimization.device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    seed = int(
        seed
        if seed is not None
        else _setting(
            config,
            "seed",
            getattr(config.optimization, "eval_seed", None)
            if getattr(config.optimization, "eval_seed", None) is not None
            else config.optimization.seed,
        )
    )
    num_steps = int(
        num_steps
        if num_steps is not None
        else _setting(config, "num_steps", config.optimization.num_steps)
    )

    totals = {
        "sq_norm": 0.0,
        "abs_norm": 0.0,
        "sq_raw": 0.0,
        "abs_raw": 0.0,
        "sq_exec_norm": 0.0,
        "abs_exec_norm": 0.0,
        "sq_exec_raw": 0.0,
        "abs_exec_raw": 0.0,
        "objective": 0.0,
    }
    counts = {
        "batches": 0,
        "windows": 0,
        "vectors": 0,
        "elements": 0,
        "exec_vectors": 0,
        "exec_elements": 0,
    }
    execute_start = int(config.task.obs_steps) - 1
    execute_end = execute_start + int(config.task.act_steps)
    modes = _module_modes(agent)
    anneal_before = _anneal_steps(agent)

    try:
        agent.eval()
        with _isolated_rng(seed), torch.inference_mode():
            for batch in loader:
                target = batch["action"][:, : int(config.task.horizon)].to(
                    config.optimization.device, non_blocking=True
                )
                if target.ndim != 3:
                    raise ValueError(
                        f"expected validation actions shaped (B,T,A), got {target.shape}"
                    )
                if execute_start < 0 or execute_end > target.shape[1]:
                    raise ValueError(
                        "executed action slice is outside the prediction horizon: "
                        f"[{execute_start}:{execute_end}] vs {target.shape[1]}"
                    )
                obs = _prepare_observations(config, batch["obs"])
                prediction = agent.sample(
                    act_0=torch.zeros_like(target),
                    obs=obs,
                    num_steps=num_steps,
                    use_ema=True,
                )
                if prediction.shape != target.shape:
                    raise ValueError(
                        "policy prediction shape does not match held-out action shape: "
                        f"{prediction.shape} vs {target.shape}"
                    )

                error = prediction - target
                exec_error = error[:, execute_start:execute_end]
                totals["sq_norm"] += torch.sum(
                    error * error, dtype=torch.float64
                ).item()
                totals["abs_norm"] += torch.sum(
                    torch.abs(error), dtype=torch.float64
                ).item()
                totals["sq_exec_norm"] += torch.sum(
                    exec_error * exec_error, dtype=torch.float64
                ).item()
                totals["abs_exec_norm"] += torch.sum(
                    torch.abs(exec_error), dtype=torch.float64
                ).item()
                totals["objective"] += torch.sum(
                    get_norm(error, str(config.optimization.norm_type)),
                    dtype=torch.float64,
                ).item()

                prediction_raw = _unnormalize(action_normalizer, prediction)
                target_raw = _unnormalize(action_normalizer, target)
                raw_error = prediction_raw - target_raw
                exec_raw_error = raw_error[:, execute_start:execute_end]
                totals["sq_raw"] += np.sum(raw_error * raw_error, dtype=np.float64)
                totals["abs_raw"] += np.sum(np.abs(raw_error), dtype=np.float64)
                totals["sq_exec_raw"] += np.sum(
                    exec_raw_error * exec_raw_error, dtype=np.float64
                )
                totals["abs_exec_raw"] += np.sum(
                    np.abs(exec_raw_error), dtype=np.float64
                )

                batch_windows, horizon, action_dim = target.shape
                counts["batches"] += 1
                counts["windows"] += int(batch_windows)
                counts["vectors"] += int(batch_windows * horizon)
                counts["elements"] += int(target.numel())
                counts["exec_vectors"] += int(
                    batch_windows * (execute_end - execute_start)
                )
                counts["exec_elements"] += int(exec_error.numel())
    finally:
        _restore_modes(modes)

    if counts["windows"] != len(dataset):
        raise RuntimeError(
            f"validated {counts['windows']} windows, expected the full {len(dataset)}"
        )
    anneal_after = _anneal_steps(agent)
    if anneal_before.keys() != anneal_after.keys() or any(
        not torch.equal(value, anneal_after[name])
        for name, value in anneal_before.items()
    ):
        raise RuntimeError("offline validation changed an encoder annealing counter")

    loss_scale = float(config.optimization.loss_scale)
    return {
        "val/objective": float(loss_scale * totals["objective"] / counts["vectors"]),
        "val/action_mse_normalized": float(totals["sq_norm"] / counts["elements"]),
        "val/action_mae_normalized": float(totals["abs_norm"] / counts["elements"]),
        "val/action_mse_raw": float(totals["sq_raw"] / counts["elements"]),
        "val/action_mae_raw": float(totals["abs_raw"] / counts["elements"]),
        "val/executed_action_mse_normalized": float(
            totals["sq_exec_norm"] / counts["exec_elements"]
        ),
        "val/executed_action_mae_normalized": float(
            totals["abs_exec_norm"] / counts["exec_elements"]
        ),
        "val/executed_action_mse_raw": float(
            totals["sq_exec_raw"] / counts["exec_elements"]
        ),
        "val/executed_action_mae_raw": float(
            totals["abs_exec_raw"] / counts["exec_elements"]
        ),
        "val/num_windows": counts["windows"],
        "val/num_batches": counts["batches"],
        "val/num_action_vectors": counts["vectors"],
        "val/num_action_elements": counts["elements"],
        "val/num_executed_action_vectors": counts["exec_vectors"],
        "val/num_executed_action_elements": counts["exec_elements"],
        "val/num_steps": num_steps,
        "val/seed": seed,
    }


def write_validation_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a complete final report atomically."""
    return _atomic_json(path, payload)


__all__ = ["validate_offline", "write_validation_report"]
