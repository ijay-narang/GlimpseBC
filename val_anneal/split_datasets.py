"""Leakage-free, trajectory-level train/validation datasets.

This module deliberately lives outside the normal training package.  It adapts the
existing image datasets for the validation-only annealing experiments without
changing their legacy behavior or the source Zarr/HDF5 data.

The public entry point is :func:`build_split_bundle`::

    bundle = build_split_bundle(
        config.task,
        "libero",
        val_fraction=0.20,
        split_seed=10000,
        manifest_path=run_dir / "split.json",
    )
    train_loader = DataLoader(bundle.train_dataset, shuffle=True, ...)
    val_loader = DataLoader(bundle.val_dataset, shuffle=False, drop_last=False, ...)

Both datasets expose the *same* ``normalizer`` object.  Its fitted values are
computed from the training trajectories only.  Splits are made before sequence
windows are enumerated, so overlapping windows can never cross partitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from mip.data.datasets.base import BaseDataset
from mip.data.datasets.libero_dataset import LiberoDataset, _demo_sort_key
from mip.data.datasets.pusht_dataset import (
    PushTImageDataset,
    download_pusht_dataset,
)
from mip.data.datasets.robomimic_dataset import RobomimicImageDataset
from mip.data.libero_utils import resolve_libero_dataset_path
from mip.data.normalizers import ImageNormalizer, MinMaxNormalizer

DEFAULT_VAL_FRACTION = 0.20
DEFAULT_SPLIT_SEED = 10000
MANIFEST_SCHEMA_VERSION = 1
SPLIT_ALGORITHM = "sorted_ids+numpy_pcg64_permutation+validation_prefix_v1"


def _natural_key(value: str) -> tuple[Any, ...]:
    """Return a deterministic numeric-aware key for trajectory names."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def _canonical_ids(trajectory_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(sorted((str(value) for value in trajectory_ids), key=_natural_key))
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory identifiers must be unique")
    if len(ids) < 2:
        raise ValueError(
            "An 80/20 train/validation split requires at least two trajectories"
        )
    return ids


@dataclass(frozen=True)
class TrajectorySplit:
    """An immutable assignment of complete trajectories to train and validation."""

    all_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    seed: int
    val_fraction: float
    algorithm: str = SPLIT_ALGORITHM

    def __post_init__(self) -> None:
        all_set = set(self.all_ids)
        train_set = set(self.train_ids)
        val_set = set(self.val_ids)
        if train_set & val_set:
            raise ValueError("Training and validation trajectory sets overlap")
        if train_set | val_set != all_set:
            raise ValueError("Training and validation sets do not exhaust trajectories")
        if not self.train_ids or not self.val_ids:
            raise ValueError(
                "Both training and validation partitions must be non-empty"
            )

    @property
    def train_count(self) -> int:
        return len(self.train_ids)

    @property
    def val_count(self) -> int:
        return len(self.val_ids)

    @property
    def id_digest(self) -> str:
        payload = "\n".join(self.all_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "trajectory_count": len(self.all_ids),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "id_sha256": self.id_digest,
            "all_ids": list(self.all_ids),
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
        }


def create_trajectory_split(
    trajectory_ids: Iterable[str],
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> TrajectorySplit:
    """Create a stable split of whole trajectories.

    ``floor(N * val_fraction)`` trajectories are held out, with a minimum of one.
    Membership is randomized by a dedicated seed and returned in canonical source
    order so data-loader order is independent of the RNG permutation.
    """
    ids = _canonical_ids(trajectory_ids)
    fraction = float(val_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")
    seed = int(split_seed)
    val_count = max(1, int(np.floor(len(ids) * fraction)))
    if val_count >= len(ids):
        raise ValueError(
            f"val_fraction={fraction} leaves no training trajectories for N={len(ids)}"
        )

    permutation = np.random.default_rng(seed).permutation(len(ids))
    val_positions = {int(index) for index in permutation[:val_count]}
    train_ids = tuple(
        value for index, value in enumerate(ids) if index not in val_positions
    )
    val_ids = tuple(value for index, value in enumerate(ids) if index in val_positions)
    return TrajectorySplit(
        all_ids=ids,
        train_ids=train_ids,
        val_ids=val_ids,
        seed=seed,
        val_fraction=fraction,
    )


def _split_from_manifest(
    payload: Mapping[str, Any],
    *,
    benchmark: str,
    expected_ids: Sequence[str],
    val_fraction: float,
    split_seed: int,
) -> TrajectorySplit:
    if int(payload.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported validation split manifest schema")
    if str(payload.get("benchmark", "")).lower() != benchmark:
        raise ValueError(
            f"Split manifest benchmark is {payload.get('benchmark')!r}, expected {benchmark!r}"
        )
    raw_split = payload.get("split")
    if not isinstance(raw_split, Mapping):
        raise ValueError("Split manifest has no `split` mapping")
    split = TrajectorySplit(
        all_ids=tuple(str(value) for value in raw_split.get("all_ids", ())),
        train_ids=tuple(str(value) for value in raw_split.get("train_ids", ())),
        val_ids=tuple(str(value) for value in raw_split.get("val_ids", ())),
        seed=int(raw_split.get("seed", -1)),
        val_fraction=float(raw_split.get("val_fraction", -1.0)),
        algorithm=str(raw_split.get("algorithm", "")),
    )
    if split.algorithm != SPLIT_ALGORITHM:
        raise ValueError(
            f"Split manifest algorithm is {split.algorithm!r}, expected {SPLIT_ALGORITHM!r}"
        )
    if split.seed != int(split_seed):
        raise ValueError(
            f"Split manifest seed is {split.seed}, requested seed is {int(split_seed)}"
        )
    if not np.isclose(split.val_fraction, float(val_fraction), rtol=0.0, atol=1e-12):
        raise ValueError(
            "Split manifest validation fraction does not match the requested fraction"
        )
    if split.all_ids != tuple(expected_ids):
        missing = sorted(set(split.all_ids) - set(expected_ids), key=_natural_key)
        added = sorted(set(expected_ids) - set(split.all_ids), key=_natural_key)
        raise ValueError(
            "Dataset trajectories differ from the saved split manifest; "
            f"missing_from_dataset={missing}, new_in_dataset={added}"
        )
    return split


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Read a split manifest without constructing a dataset."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Split manifest must contain a JSON object: {manifest_path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _source_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    provenance: dict[str, Any] = {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if resolved.is_file():
        provenance["size_bytes"] = int(stat.st_size)
    return provenance


def _manifest_payload(
    *,
    benchmark: str,
    split: TrajectorySplit,
    source_path: Path,
    trajectory_lengths: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": "offline_validation_no_rollouts",
        "benchmark": benchmark,
        "source": _source_provenance(source_path),
        "trajectory_lengths": {
            key: int(trajectory_lengths[key]) for key in split.all_ids
        },
        "split": split.as_dict(),
        "invariants": {
            "whole_trajectory_split": True,
            "train_only_normalization": True,
            "shared_normalizer_object": True,
            "validation_shuffle": False,
            "validation_drop_last": False,
        },
    }


def _load_or_create_split(
    *,
    benchmark: str,
    trajectory_ids: Sequence[str],
    val_fraction: float,
    split_seed: int,
    manifest_path: Path | None,
    source_path: Path,
    trajectory_lengths: Mapping[str, int],
) -> tuple[TrajectorySplit, dict[str, Any]]:
    expected_ids = _canonical_ids(trajectory_ids)
    if manifest_path is not None and manifest_path.exists():
        payload = load_split_manifest(manifest_path)
        split = _split_from_manifest(
            payload,
            benchmark=benchmark,
            expected_ids=expected_ids,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        saved_lengths = payload.get("trajectory_lengths", {})
        if saved_lengths:
            current_lengths = {
                key: int(trajectory_lengths[key]) for key in split.all_ids
            }
            normalized_saved = {
                str(key): int(value) for key, value in saved_lengths.items()
            }
            if normalized_saved != current_lengths:
                raise ValueError(
                    "Trajectory lengths differ from the saved split manifest"
                )
        saved_source = payload.get("source")
        current_source = _source_provenance(source_path)
        if saved_source != current_source:
            raise ValueError(
                "Dataset source provenance differs from the saved split manifest"
            )
        return split, payload

    split = create_trajectory_split(
        expected_ids,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    payload = _manifest_payload(
        benchmark=benchmark,
        split=split,
        source_path=source_path,
        trajectory_lengths=trajectory_lengths,
    )
    if manifest_path is not None:
        _atomic_write_json(manifest_path, payload)
    return split, payload


class _ConfigOverlay:
    """Read-only attribute overlay that does not mutate an OmegaConf object."""

    def __init__(self, base: Any, **overrides: Any):
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


class _PushTImageWithoutNormalizer(PushTImageDataset):
    """Construct the existing PushT reader without fitting full-data statistics."""

    def get_normalizer(self, **kwargs):
        return None


@contextmanager
def _zarr_require_dataset_compatibility():
    """Bridge the legacy Robomimic converter to the installed Zarr 3 API.

    The repository converter calls ``Group.require_dataset``.  Zarr 3.2 removed
    that compatibility spelling in favor of the otherwise equivalent
    ``require_array``.  Keep the bridge local to construction of these isolated
    datasets instead of changing the existing converter or globally retaining a
    monkey patch for the rest of the process.
    """
    import zarr

    group_type = type(zarr.group(store=zarr.storage.MemoryStore()))
    if hasattr(group_type, "require_dataset"):
        yield
        return

    def require_dataset(group, name, *, shape, **kwargs):
        return group.require_array(name, shape=shape, **kwargs)

    group_type.require_dataset = require_dataset
    try:
        yield
    finally:
        if getattr(group_type, "require_dataset", None) is require_dataset:
            delattr(group_type, "require_dataset")


class _RobomimicImageWithNormalizer(RobomimicImageDataset):
    """Allow validation construction with an already-fitted train normalizer."""

    def __init__(self, *args: Any, normalizer: Any = None, **kwargs: Any):
        self._provided_normalizer = normalizer
        with _zarr_require_dataset_compatibility():
            super().__init__(*args, **kwargs)

    def get_normalizer(self):
        if self._provided_normalizer is not None:
            return self._provided_normalizer
        return super().get_normalizer()


class _LiberoValidationDataset(LiberoDataset):
    """LIBERO validation reader that never fits statistics on held-out data."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        template: LiberoDataset,
        expected_ids: Sequence[str],
        normalizer: Any,
    ):
        BaseDataset.__init__(self)
        self.dataset_path = str(Path(dataset_path).resolve())
        self.shape_meta = template.shape_meta
        self.horizon = template.horizon
        self.obs_steps = template.obs_steps
        self.pad_before = template.pad_before
        self.pad_after = template.pad_after
        self.action_dim = template.action_dim
        self.rgb_keys = list(template.rgb_keys)
        self.lowdim_keys = list(template.lowdim_keys)
        self.source_obs_keys = dict(template.source_obs_keys)
        self._h5_file = None
        self.samples: list[tuple[str, int]] = []
        self.normalizer = normalizer

        with h5py.File(self.dataset_path, "r") as dataset_file:
            if "data" not in dataset_file:
                raise ValueError("LIBERO validation view has no `data` group")
            self.demos = sorted(dataset_file["data"].keys(), key=_demo_sort_key)
            if tuple(self.demos) != tuple(expected_ids):
                raise ValueError("LIBERO validation view does not match split IDs")
            for demo_name in self.demos:
                demo = dataset_file[f"data/{demo_name}"]
                if "actions" not in demo or "obs" not in demo:
                    raise ValueError(
                        f"Demonstration {demo_name!r} must contain `actions` and `obs`"
                    )
                actions = demo["actions"]
                if actions.ndim != 2 or int(actions.shape[-1]) != self.action_dim:
                    raise ValueError(
                        f"{demo_name}/actions has shape {actions.shape}; expected "
                        f"(T, {self.action_dim})"
                    )
                episode_length = int(actions.shape[0])
                if episode_length <= 0:
                    raise ValueError(f"Validation demonstration {demo_name!r} is empty")
                for key in self.rgb_keys + self.lowdim_keys:
                    source_key = self.source_obs_keys[key]
                    if source_key not in demo["obs"]:
                        raise ValueError(
                            f"{demo_name}/obs is missing required key {source_key!r}"
                        )
                    if int(demo["obs"][source_key].shape[0]) != episode_length:
                        raise ValueError(
                            f"{demo_name}/obs/{source_key} length does not match actions"
                        )
                min_start = -self.pad_before
                max_start = episode_length - self.horizon + self.pad_after
                self.samples.extend(
                    (demo_name, start) for start in range(min_start, max_start + 1)
                )
        if not self.samples:
            raise ValueError(
                "No sampleable LIBERO validation trajectories were selected"
            )


class TrajectoryDatasetView(Dataset):
    """Index view over a dataset with explicit trajectory provenance."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        trajectory_ids: Sequence[str],
        *,
        mode: str,
        normalizer: Any,
    ):
        self.base_dataset = base_dataset
        self.indices = tuple(int(index) for index in indices)
        self.trajectory_ids = tuple(str(value) for value in trajectory_ids)
        self.mode = mode
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]

    def __getattr__(self, name: str) -> Any:
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __str__(self) -> str:
        return (
            f"{self.mode} trajectories={len(self.trajectory_ids)} "
            f"sequences={len(self)} base=({self.base_dataset})"
        )


@dataclass
class SplitBundle:
    """Datasets and provenance for one offline-validation training run."""

    train_dataset: Dataset
    val_dataset: Dataset
    normalizer: Any
    split: TrajectorySplit
    benchmark: str
    source_path: Path
    manifest: dict[str, Any]
    manifest_path: Path | None = None
    _temporary_directory: tempfile.TemporaryDirectory | None = field(
        default=None, repr=False
    )

    @property
    def train_count(self) -> int:
        return self.split.train_count

    @property
    def val_count(self) -> int:
        return self.split.val_count

    @property
    def train_sequence_count(self) -> int:
        return len(self.train_dataset)

    @property
    def val_sequence_count(self) -> int:
        return len(self.val_dataset)

    @property
    def dataset_path(self) -> str:
        """Expose the immutable source path to reproducibility metadata."""
        return str(self.source_path)

    @property
    def train_ids(self) -> tuple[str, ...]:
        return self.split.train_ids

    @property
    def val_ids(self) -> tuple[str, ...]:
        return self.split.val_ids

    def close(self) -> None:
        """Close lazy readers and remove temporary read-only HDF5 views."""
        seen: set[int] = set()
        for dataset in (self.train_dataset, self.val_dataset):
            base = getattr(dataset, "base_dataset", dataset)
            if id(base) in seen:
                continue
            seen.add(id(base))
            close = getattr(base, "close", None)
            if callable(close):
                close()
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> SplitBundle:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _resolve_pusht_path(task_config: Any) -> Path:
    explicit = getattr(task_config, "dataset_path", None)
    if explicit:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit))))
    else:
        repo = getattr(task_config, "dataset_repo", None)
        filename = getattr(task_config, "dataset_filename", None)
        if not repo or not filename:
            raise ValueError(
                "PushT requires dataset_path or dataset_repo/dataset_filename"
            )
        if str(filename).endswith(".zip"):
            path = Path(download_pusht_dataset(str(filename), str(repo)))
        else:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=str(repo), filename=str(filename), repo_type="dataset"
                )
            )
    if not path.exists():
        raise FileNotFoundError(f"PushT dataset not found: {path}")
    return path.resolve()


def _resolve_robomimic_path(task_config: Any) -> Path:
    explicit = getattr(task_config, "dataset_path", None)
    if explicit:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit))))
    else:
        repo = getattr(task_config, "dataset_repo", None)
        filename = getattr(task_config, "dataset_filename", None)
        if not repo or not filename:
            raise ValueError(
                "Robomimic requires dataset_path or dataset_repo/dataset_filename"
            )
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=str(repo), filename=str(filename), repo_type="dataset"
            )
        )
    if not path.is_file():
        raise FileNotFoundError(f"Robomimic dataset not found: {path}")
    return path.resolve()


def _hdf5_trajectory_inventory(
    path: Path,
    *,
    filter_key: str | None = None,
) -> tuple[tuple[str, ...], dict[str, int]]:
    with h5py.File(path, "r") as dataset_file:
        if "data" not in dataset_file:
            raise ValueError(f"HDF5 dataset has no `data` group: {path}")
        if filter_key:
            mask_path = f"mask/{filter_key}"
            if mask_path not in dataset_file:
                raise ValueError(f"HDF5 dataset has no filter mask {mask_path!r}")
            raw_ids = dataset_file[mask_path][()]
            ids = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in raw_ids
            ]
        else:
            ids = list(dataset_file["data"].keys())
        ids_tuple = _canonical_ids(ids)
        lengths: dict[str, int] = {}
        for trajectory_id in ids_tuple:
            if trajectory_id not in dataset_file["data"]:
                raise ValueError(
                    f"Filter references missing demonstration {trajectory_id!r}"
                )
            demo = dataset_file[f"data/{trajectory_id}"]
            if "actions" not in demo:
                raise ValueError(f"{trajectory_id!r} has no actions dataset")
            length = int(demo["actions"].shape[0])
            if length <= 0:
                raise ValueError(f"Trajectory {trajectory_id!r} is empty")
            lengths[trajectory_id] = length
    return ids_tuple, lengths


def _write_hdf5_link_view(
    source_path: Path,
    destination_path: Path,
    trajectory_ids: Sequence[str],
    *,
    renumber: bool,
) -> None:
    """Create a tiny external-link view; never mutate or duplicate source data."""
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(destination_path, "w") as destination,
    ):
        for key, value in source.attrs.items():
            destination.attrs[key] = value
        data = destination.create_group("data")
        for index, source_id in enumerate(trajectory_ids):
            if source_id not in source["data"]:
                raise ValueError(f"Source dataset has no trajectory {source_id!r}")
            destination_id = f"demo_{index}" if renumber else source_id
            data[destination_id] = h5py.ExternalLink(
                str(source_path.resolve()), f"/data/{source_id}"
            )


def _temporary_views(
    *, manifest_path: Path | None, benchmark: str
) -> tempfile.TemporaryDirectory:
    parent: Path | None = None
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        parent = manifest_path.parent
    return tempfile.TemporaryDirectory(
        prefix=f".{benchmark}_split_views_",
        dir=str(parent) if parent is not None else None,
    )


def _build_pusht_bundle(
    task_config: Any,
    *,
    val_fraction: float,
    split_seed: int,
    manifest_path: Path | None,
) -> SplitBundle:
    if str(getattr(task_config, "obs_type", "")) != "image":
        raise ValueError("Annealed validation supports only the PushT image dataset")
    path = _resolve_pusht_path(task_config)
    dataset = _PushTImageWithoutNormalizer(
        dataset_path=str(path),
        shape_meta=task_config.shape_meta,
        n_obs_steps=int(task_config.obs_steps),
        horizon=int(task_config.horizon),
        pad_before=int(task_config.obs_steps) - 1,
        pad_after=int(task_config.act_steps) - 1,
    )
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends, dtype=np.int64)
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    ids = tuple(f"episode_{index}" for index in range(len(episode_ends)))
    lengths = {
        trajectory_id: int(episode_ends[index] - episode_starts[index])
        for index, trajectory_id in enumerate(ids)
    }
    split, manifest = _load_or_create_split(
        benchmark="pusht",
        trajectory_ids=ids,
        val_fraction=val_fraction,
        split_seed=split_seed,
        manifest_path=manifest_path,
        source_path=path,
        trajectory_lengths=lengths,
    )

    id_to_episode = {trajectory_id: index for index, trajectory_id in enumerate(ids)}
    train_episode_indices = [id_to_episode[value] for value in split.train_ids]

    def selected_values(key: str, suffix: slice | None = None) -> np.ndarray:
        array = dataset.replay_buffer[key]
        pieces = []
        for episode_index in train_episode_indices:
            start = int(episode_starts[episode_index])
            end = int(episode_ends[episode_index])
            piece = np.asarray(array[start:end])
            if suffix is not None:
                piece = piece[..., suffix]
            pieces.append(piece)
        return np.concatenate(pieces, axis=0)

    normalizer = defaultdict(dict)
    normalizer["obs"]["image"] = ImageNormalizer()
    normalizer["obs"]["agent_pos"] = MinMaxNormalizer(
        selected_values("state", slice(0, 2))
    )
    normalizer["action"] = MinMaxNormalizer(selected_values("action"))
    dataset.normalizer = normalizer

    sample_episode_indices = np.searchsorted(
        episode_ends,
        np.asarray(dataset.sampler.indices)[:, 0],
        side="right",
    )
    train_episodes = set(train_episode_indices)
    train_indices = [
        index
        for index, episode_index in enumerate(sample_episode_indices)
        if int(episode_index) in train_episodes
    ]
    val_indices = [
        index
        for index, episode_index in enumerate(sample_episode_indices)
        if int(episode_index) not in train_episodes
    ]
    train_dataset = TrajectoryDatasetView(
        dataset,
        train_indices,
        split.train_ids,
        mode="train",
        normalizer=normalizer,
    )
    val_dataset = TrajectoryDatasetView(
        dataset,
        val_indices,
        split.val_ids,
        mode="val",
        normalizer=normalizer,
    )
    return SplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        normalizer=normalizer,
        split=split,
        benchmark="pusht",
        source_path=path,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def _build_libero_bundle(
    task_config: Any,
    *,
    val_fraction: float,
    split_seed: int,
    manifest_path: Path | None,
) -> SplitBundle:
    if str(getattr(task_config, "obs_type", "")) != "image":
        raise ValueError("Annealed validation supports only the LIBERO image dataset")
    path = Path(resolve_libero_dataset_path(task_config)).resolve()
    filter_key = getattr(task_config, "dataset_filter_key", None)
    ids, lengths = _hdf5_trajectory_inventory(path, filter_key=filter_key)
    split, manifest = _load_or_create_split(
        benchmark="libero",
        trajectory_ids=ids,
        val_fraction=val_fraction,
        split_seed=split_seed,
        manifest_path=manifest_path,
        source_path=path,
        trajectory_lengths=lengths,
    )

    temporary_directory = _temporary_views(
        manifest_path=manifest_path, benchmark="libero"
    )
    temporary_root = Path(temporary_directory.name)
    train_path = temporary_root / "train.hdf5"
    val_path = temporary_root / "val.hdf5"
    _write_hdf5_link_view(path, train_path, split.train_ids, renumber=False)
    _write_hdf5_link_view(path, val_path, split.val_ids, renumber=False)
    split_config = _ConfigOverlay(
        task_config,
        dataset_path=str(train_path),
        dataset_repo=None,
        dataset_filename=None,
        dataset_filter_key=None,
        val_dataset_percentage=0.0,
    )
    try:
        train_dataset = LiberoDataset(train_path, split_config, mode="train")
        normalizer = train_dataset.normalizer
        val_dataset = _LiberoValidationDataset(
            val_path,
            template=train_dataset,
            expected_ids=split.val_ids,
            normalizer=normalizer,
        )
        train_dataset.trajectory_ids = split.train_ids
        val_dataset.trajectory_ids = split.val_ids
    except Exception:
        temporary_directory.cleanup()
        raise
    if val_dataset.normalizer is not train_dataset.normalizer:
        raise AssertionError("LIBERO validation did not receive the train normalizer")
    return SplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        normalizer=normalizer,
        split=split,
        benchmark="libero",
        source_path=path,
        manifest=manifest,
        manifest_path=manifest_path,
        _temporary_directory=temporary_directory,
    )


def _build_robomimic_bundle(
    task_config: Any,
    *,
    val_fraction: float,
    split_seed: int,
    manifest_path: Path | None,
) -> SplitBundle:
    if str(getattr(task_config, "obs_type", "")) != "image":
        raise ValueError(
            "Annealed validation supports only the Robomimic image dataset"
        )
    path = _resolve_robomimic_path(task_config)
    ids, lengths = _hdf5_trajectory_inventory(path)
    split, manifest = _load_or_create_split(
        benchmark="robomimic",
        trajectory_ids=ids,
        val_fraction=val_fraction,
        split_seed=split_seed,
        manifest_path=manifest_path,
        source_path=path,
        trajectory_lengths=lengths,
    )

    temporary_directory = _temporary_views(
        manifest_path=manifest_path, benchmark="robomimic"
    )
    temporary_root = Path(temporary_directory.name)
    train_path = temporary_root / "train.hdf5"
    val_path = temporary_root / "val.hdf5"
    _write_hdf5_link_view(path, train_path, split.train_ids, renumber=True)
    _write_hdf5_link_view(path, val_path, split.val_ids, renumber=True)
    common = {
        "shape_meta": task_config.shape_meta,
        "n_obs_steps": int(task_config.obs_steps),
        "horizon": int(task_config.horizon),
        "pad_before": int(task_config.obs_steps) - 1,
        "pad_after": int(task_config.act_steps) - 1,
        "abs_action": bool(task_config.abs_action),
        "val_dataset_percentage": 0.0,
    }
    try:
        train_dataset = _RobomimicImageWithNormalizer(
            str(train_path), mode="train", **common
        )
        normalizer = train_dataset.normalizer
        val_dataset = _RobomimicImageWithNormalizer(
            str(val_path), mode="val", normalizer=normalizer, **common
        )
        train_dataset.trajectory_ids = split.train_ids
        val_dataset.trajectory_ids = split.val_ids
    except Exception:
        temporary_directory.cleanup()
        raise
    if val_dataset.normalizer is not train_dataset.normalizer:
        raise AssertionError(
            "Robomimic validation did not receive the train normalizer"
        )
    return SplitBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        normalizer=normalizer,
        split=split,
        benchmark="robomimic",
        source_path=path,
        manifest=manifest,
        manifest_path=manifest_path,
        _temporary_directory=temporary_directory,
    )


def build_split_bundle(
    task_config: Any,
    benchmark: str,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    manifest_path: str | Path | None = None,
) -> SplitBundle:
    """Build train and validation image datasets for an annealing job.

    Args:
        task_config: Existing Hydra task configuration.
        benchmark: One of ``pusht``, ``libero``, or ``robomimic``.
        val_fraction: Fraction of complete trajectories held out for validation.
        split_seed: Dedicated seed controlling only trajectory membership.
        manifest_path: Optional durable JSON path. If it already exists its exact
            assignment is reused after validating the source IDs and lengths.

    Returns:
        A :class:`SplitBundle`. Call ``bundle.close()`` after all DataLoader workers
        have stopped, or use it as a context manager.
    """
    normalized_benchmark = str(benchmark).strip().lower().replace("-", "_")
    aliases = {
        "push_t": "pusht",
        "pusht": "pusht",
        "libero": "libero",
        "robomimic": "robomimic",
    }
    if normalized_benchmark not in aliases:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}; choose pusht, libero, or robomimic"
        )
    normalized_benchmark = aliases[normalized_benchmark]
    resolved_manifest = Path(manifest_path).resolve() if manifest_path else None
    builders = {
        "pusht": _build_pusht_bundle,
        "libero": _build_libero_bundle,
        "robomimic": _build_robomimic_bundle,
    }
    return builders[normalized_benchmark](
        task_config,
        val_fraction=float(val_fraction),
        split_seed=int(split_seed),
        manifest_path=resolved_manifest,
    )


__all__ = [
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VAL_FRACTION",
    "SplitBundle",
    "TrajectoryDatasetView",
    "TrajectorySplit",
    "build_split_bundle",
    "create_trajectory_split",
    "load_split_manifest",
]
