"""Opt-in reproducible sampling and local experiment provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


def seed_worker(worker_id):
    """Derive NumPy/Python worker seeds from the DataLoader's Torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_seed_kwargs(config):
    """Isolate shuffled sample order from policy/evaluation RNG consumption."""
    if not getattr(config.optimization, "deterministic", False):
        return {}
    return {
        "generator": torch.Generator().manual_seed(config.optimization.seed),
        "worker_init_fn": seed_worker,
    }


def evaluation_seed(config):
    """Use an explicit evaluation seed, falling back to the training seed."""
    seed = getattr(config.optimization, "eval_seed", None)
    seed = int(config.optimization.seed if seed is None else seed)
    if not 0 <= seed < 2**32:
        raise ValueError("eval_seed must be between 0 and 4294967295")
    return seed


def reproducible_evaluation(function):
    """Repeat evaluation RNG draws without advancing training RNGs in strict mode."""

    @wraps(function)
    def wrapped(config, *args, **kwargs):
        if not getattr(config.optimization, "deterministic", False):
            return function(config, *args, **kwargs)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_initialized()
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            try:
                seed = evaluation_seed(config)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                return function(config, *args, **kwargs)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    return wrapped


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def record_run_metadata(config, dataset=None):
    """Save the effective config, software versions, and available source state.

    This is provenance for fresh runs, not an RNG/sampler resume checkpoint.
    Dataset contents and third-party source/weights are not archived here.
    """
    root = Path(__file__).resolve().parents[1]
    # LIBERO's isolated environment uses Python 3.10.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017
    output = (
        Path(config.log.log_dir) / "reproducibility" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    resolved = OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), output / "config.yaml", resolve=True)
    packages = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
        }
    )
    (output / "packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    source_hashes = {}
    # Restrict the archive to first-party code/config/scripts and dependency locks.
    # Include nonignored untracked files so a dirty development run is traceable.
    names = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if names is not None:
        with zipfile.ZipFile(
            output / "source.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for name in sorted(set(names.split("\0")) - {""}):
                relative = Path(name)
                source = root / relative
                allowed = name in {"pyproject.toml", "uv.lock"} or (
                    relative.parts[0]
                    in {"mip", "examples", "scripts", "glitr"}
                    and relative.suffix in {".py", ".yaml", ".yml", ".sh", ".toml"}
                )
                if allowed and source.is_file() and not source.is_symlink():
                    content = source.read_bytes()
                    source_hashes[name] = hashlib.sha256(content).hexdigest()
                    archive.writestr(name, content)
    dataset_path = getattr(dataset, "dataset_path", None) or getattr(
        config.task, "dataset_path", None
    )
    metadata = {
        "created_utc": stamp,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MUJOCO_GL",
                "SLURM_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
            )
        },
        "git_commit": (_git(root, "rev-parse", "HEAD") or "").strip() or None,
        "git_status": _git(root, "status", "--short"),
        "git_submodules": _git(root, "submodule", "status", "--recursive"),
        "source_sha256": source_hashes,
        "dataset_path": str(Path(dataset_path).expanduser().resolve())
        if dataset_path
        else None,
        "dataset_contents_hashed": False,
        "exact_resume_supported": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved reproducibility metadata to {}", output)
    return output
