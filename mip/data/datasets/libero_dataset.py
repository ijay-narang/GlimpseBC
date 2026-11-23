"""Lazy HDF5 dataset adapter for one official LIBERO manipulation task."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from mip.data.datasets.base import BaseDataset
from mip.data.libero_utils import resolve_libero_dataset_path
from mip.data.normalizers import ImageNormalizer, MinMaxNormalizer


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def _select_demos(
    dataset_file: h5py.File,
    *,
    mode: str,
    val_fraction: float,
    filter_key: str | None,
) -> list[str]:
    if "data" not in dataset_file:
        raise ValueError("LIBERO HDF5 file is missing the required `data` group")
    if filter_key:
        mask_path = f"mask/{filter_key}"
        if mask_path not in dataset_file:
            raise ValueError(f"LIBERO HDF5 file has no filter mask {mask_path!r}")
        demos = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in dataset_file[mask_path][()]
        ]
    else:
        demos = list(dataset_file["data"].keys())
    demos = sorted(demos, key=_demo_sort_key)
    if not demos:
        raise ValueError("LIBERO HDF5 file contains no demonstrations")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_dataset_percentage must be in [0, 1)")
    if mode not in {"train", "val"}:
        raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

    val_count = int(len(demos) * val_fraction)
    if mode == "val":
        if val_count == 0:
            raise ValueError(
                "Validation mode requested but val_dataset_percentage selects no demos"
            )
        return demos[-val_count:]
    return demos[:-val_count] if val_count else demos


def _read_window(dataset: h5py.Dataset, start: int, length: int) -> np.ndarray:
    """Read a logical window and repeat boundary values for padding."""
    if length <= 0:
        raise ValueError(f"Window length must be positive, got {length}")
    episode_length = int(dataset.shape[0])
    if episode_length <= 0:
        raise ValueError(f"Cannot sample empty HDF5 dataset {dataset.name}")
    logical_indices = np.arange(start, start + length, dtype=np.int64)
    clamped = np.clip(logical_indices, 0, episode_length - 1)
    first = int(clamped.min())
    last = int(clamped.max()) + 1
    values = dataset[first:last]
    return np.asarray(values[clamped - first])


class LiberoDataset(BaseDataset):
    """One-task LIBERO behavior-cloning dataset.

    Image arrays stay on disk and are opened independently in each DataLoader worker.
    Only action and low-dimensional statistics are loaded during initialization.
    """

    def __init__(self, dataset_path: str | Path, task_config: Any, mode: str = "train"):
        super().__init__()
        self.dataset_path = str(Path(dataset_path).resolve())
        self.shape_meta = task_config.shape_meta
        self.horizon = int(task_config.horizon)
        self.obs_steps = int(task_config.obs_steps)
        self.pad_before = self.obs_steps - 1
        self.pad_after = int(task_config.act_steps) - 1
        self.action_dim = int(task_config.act_dim)
        self._h5_file: h5py.File | None = None

        self.rgb_keys: list[str] = []
        self.lowdim_keys: list[str] = []
        for key, attr in self.shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(str(key))
            elif obs_type == "low_dim":
                self.lowdim_keys.append(str(key))
            else:
                raise ValueError(f"Unsupported LIBERO observation type {obs_type!r}")
        if not self.rgb_keys:
            raise ValueError(
                "The initial LIBERO pipeline requires at least one RGB key"
            )
        self.source_obs_keys = {key: key for key in self.rgb_keys + self.lowdim_keys}

        val_fraction = float(getattr(task_config, "val_dataset_percentage", 0.0))
        filter_key = getattr(task_config, "dataset_filter_key", None)
        self.samples: list[tuple[str, int]] = []
        action_stats: list[np.ndarray] = []
        lowdim_stats: dict[str, list[np.ndarray]] = {
            key: [] for key in self.lowdim_keys
        }

        with h5py.File(self.dataset_path, "r") as dataset_file:
            self.demos = _select_demos(
                dataset_file,
                mode=mode,
                val_fraction=val_fraction,
                filter_key=filter_key,
            )
            for demo_name in self.demos:
                demo = dataset_file[f"data/{demo_name}"]
                if "actions" not in demo or "obs" not in demo:
                    raise ValueError(
                        f"Demonstration {demo_name!r} must contain `actions` and `obs`"
                    )
                actions = np.asarray(demo["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"{demo_name}/actions has shape {actions.shape}; expected "
                        f"(T, {self.action_dim})"
                    )
                episode_length = int(actions.shape[0])
                if episode_length == 0:
                    continue
                for key in self.rgb_keys + self.lowdim_keys:
                    source_key = self.source_obs_keys[key]
                    if source_key not in demo["obs"]:
                        available = ", ".join(sorted(demo["obs"].keys()))
                        raise ValueError(
                            f"{demo_name}/obs is missing {source_key!r} for model key "
                            f"{key!r}; available: {available}."
                        )
                    if int(demo["obs"][source_key].shape[0]) != episode_length:
                        raise ValueError(
                            f"{demo_name}/obs/{source_key} length does not match actions"
                        )
                    if key in self.rgb_keys:
                        actual_shape = tuple(
                            int(value) for value in demo["obs"][source_key].shape[1:]
                        )
                        expected_shape = tuple(
                            int(value) for value in self.shape_meta["obs"][key]["shape"]
                        )
                        expected_hwc = (
                            expected_shape[-2],
                            expected_shape[-1],
                            expected_shape[0],
                        )
                        if actual_shape not in {expected_shape, expected_hwc}:
                            raise ValueError(
                                f"{demo_name}/obs/{source_key} has frame shape "
                                f"{actual_shape}; expected {expected_hwc} (HWC) or "
                                f"{expected_shape} (CHW)"
                            )
                action_stats.append(actions)
                for key in self.lowdim_keys:
                    lowdim_stats[key].append(
                        np.asarray(
                            demo["obs"][self.source_obs_keys[key]],
                            dtype=np.float32,
                        )
                    )

                min_start = -self.pad_before
                max_start = episode_length - self.horizon + self.pad_after
                self.samples.extend(
                    (demo_name, start) for start in range(min_start, max_start + 1)
                )

        if not self.samples or not action_stats:
            raise ValueError("No sampleable LIBERO demonstrations were selected")

        self.normalizer = defaultdict(dict)
        for key in self.rgb_keys:
            self.normalizer["obs"][key] = ImageNormalizer()
        for key in self.lowdim_keys:
            self.normalizer["obs"][key] = MinMaxNormalizer(
                np.concatenate(lowdim_stats[key], axis=0)
            )
        self.normalizer["action"] = MinMaxNormalizer(
            np.concatenate(action_stats, axis=0)
        )

    def _get_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_path, "r", swmr=True)
        return self._h5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def close(self) -> None:
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return (
            f"LIBERO file={self.dataset_path} demos={len(self.demos)} "
            f"sequences={len(self)} camera_fov={self.camera_fov}"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        demo_name, start = self.samples[idx]
        demo = self._get_file()[f"data/{demo_name}"]
        obs: dict[str, torch.Tensor] = {}

        for key in self.rgb_keys:
            source_key = self.source_obs_keys[key]
            image = _read_window(demo["obs"][source_key], start, self.obs_steps)
            expected_channels = int(self.shape_meta["obs"][key]["shape"][0])
            if image.shape[-1] == expected_channels:
                image = np.moveaxis(image, -1, -3)
            elif image.shape[-3] != expected_channels:
                raise ValueError(
                    f"{demo_name}/obs/{source_key} has unsupported image shape {image.shape}"
                )
            image = image.astype(np.float32)
            if (
                np.issubdtype(demo["obs"][source_key].dtype, np.integer)
                or image.max() > 1.0
            ):
                image /= 255.0
            image = self.normalizer["obs"][key].normalize(image)
            obs[key] = torch.from_numpy(np.ascontiguousarray(image))

        for key in self.lowdim_keys:
            value = _read_window(
                demo["obs"][self.source_obs_keys[key]], start, self.obs_steps
            ).astype(np.float32)
            value = self.normalizer["obs"][key].normalize(value)
            obs[key] = torch.from_numpy(np.ascontiguousarray(value))

        action = _read_window(demo["actions"], start, self.horizon).astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        return {
            "obs": obs,
            "action": torch.from_numpy(np.ascontiguousarray(action)),
        }


def make_dataset(task_config: Any, mode: str = "train") -> LiberoDataset:
    """Build a LIBERO dataset without importing the simulator when path is explicit."""
    dataset_path = resolve_libero_dataset_path(task_config)
    return LiberoDataset(dataset_path=dataset_path, task_config=task_config, mode=mode)


# Preserve legacy pickle paths while implementations live in the data package.
