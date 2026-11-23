"""LIBERO-specific discovery helpers with no effect on other task pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_LIBERO_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
}


def get_libero_suite_name(task_config: Any) -> str:
    """Return and validate the configured official LIBERO suite name."""
    suite_name = getattr(task_config, "libero_suite", None)
    if suite_name is None:
        suite_name = getattr(task_config, "env_type", None)
    suite_name = str(suite_name).lower()
    if suite_name not in SUPPORTED_LIBERO_SUITES:
        choices = ", ".join(sorted(SUPPORTED_LIBERO_SUITES))
        raise ValueError(
            f"Unsupported LIBERO suite {suite_name!r}; choose one of {choices}"
        )
    return suite_name


def get_libero_task_id(task_config: Any) -> int:
    """Return the configured task index."""
    task_id = int(getattr(task_config, "libero_task_id", 0))
    if task_id < 0:
        raise ValueError(f"libero_task_id must be non-negative, got {task_id}")
    return task_id


def import_libero_benchmark():
    """Import LIBERO lazily and provide an actionable dependency error."""
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise ImportError(
            "LIBERO is required only for LIBERO environment creation and automatic "
            "task-path discovery. Set up the isolated environment with "
            "`bash scripts/libero/utils/setup_libero_env.sh`, or provide "
            "task.dataset_path "
            "when using the dataset loader by itself."
        ) from exc
    return benchmark


def make_libero_suite(task_config: Any):
    """Instantiate the configured official benchmark suite."""
    benchmark = import_libero_benchmark()
    suite_name = get_libero_suite_name(task_config)
    task_order_index = int(getattr(task_config, "libero_task_order_index", 0))
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict))
        raise ValueError(
            f"LIBERO installation does not provide {suite_name!r}; available: {available}"
        )
    suite = benchmark_dict[suite_name](task_order_index=task_order_index)
    task_id = get_libero_task_id(task_config)
    if task_id >= suite.n_tasks:
        raise ValueError(
            f"libero_task_id={task_id} is out of range for {suite_name} "
            f"({suite.n_tasks} tasks)"
        )
    return suite


def get_libero_init_states_path(task: Any) -> Path:
    """Resolve one task's fixed-initial-state file from official LIBERO."""
    from libero.libero import get_libero_path

    return (
        Path(get_libero_path("init_states"))
        / str(task.problem_folder)
        / str(task.init_states_file)
    )


def load_libero_init_states(task: Any):
    """Load trusted fixed initial states from the pinned official LIBERO source."""
    import torch

    return torch.load(get_libero_init_states_path(task), weights_only=False)


def resolve_libero_dataset_path(task_config: Any) -> Path:
    """Resolve an HDF5 demonstration file from explicit or official locations."""
    explicit_path = getattr(task_config, "dataset_path", None)
    if explicit_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit_path))))
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO dataset file not found: {path}")
        return path

    dataset_repo = getattr(task_config, "dataset_repo", None)
    dataset_filename = getattr(task_config, "dataset_filename", None)
    if dataset_repo and dataset_filename:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=str(dataset_repo),
                filename=str(dataset_filename),
                repo_type="dataset",
            )
        )

    suite = make_libero_suite(task_config)
    task_id = get_libero_task_id(task_config)
    relative_path = Path(suite.get_task_demonstration(task_id))

    dataset_root = getattr(task_config, "dataset_root", None) or os.environ.get(
        "LIBERO_DATA_ROOT"
    )
    if dataset_root is None:
        try:
            from libero.libero import get_libero_path
        except ImportError as exc:
            raise ImportError(
                "Set task.dataset_path, task.dataset_root, or LIBERO_DATA_ROOT to "
                "locate LIBERO demonstrations."
            ) from exc
        dataset_root = get_libero_path("datasets")

    path = (
        Path(os.path.expandvars(os.path.expanduser(str(dataset_root)))) / relative_path
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"LIBERO dataset file not found: {path}. Download it with "
            "`bash scripts/libero/utils/libero_fetch_data.sh`."
        )
    return path


# Preserve legacy pickle paths while implementations live in the data package.
