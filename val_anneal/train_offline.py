"""Environment-free train/validation entrypoint for annealing sweeps.

The legacy benchmark modules remain untouched.  This entrypoint builds an
episode-level 80/20 split, calls their existing ``train`` function with
``envs=None`` after making rollout evaluation unreachable, and then performs one
complete offline validation pass.
"""

# ruff: noqa: E402, I001 -- repo root must precede first-party imports when run as a file.

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import loguru
import numpy as np

from mip.agent import TrainingAgent
from mip.flop_accounting import get_job_flop_metrics, report_job_flops
from mip.glitr import make_agent, prepare_glitr
from mip.logger import Logger
from mip.reproducibility import record_run_metadata
from mip.torch_utils import limit_threads, set_seed
from val_anneal.validate import validate_offline, write_validation_report


_TRAIN_MODULES = {
    "pusht": "examples.train_pusht",
    "libero": "examples.train_libero",
    "robomimic": "examples.train_robomimic",
}


def _validation_setting(config: Any, name: str, default: Any) -> Any:
    section = getattr(config, "validation", None)
    return getattr(section, name, default) if section is not None else default


def _infer_benchmark(config: Any) -> str:
    explicit = _validation_setting(config, "benchmark", None)
    if explicit is not None:
        benchmark = str(explicit).lower()
    elif getattr(config.task, "libero_suite", None) is not None:
        benchmark = "libero"
    elif str(config.task.env_name).lower() == "pusht":
        benchmark = "pusht"
    else:
        benchmark = "robomimic"
    if benchmark not in _TRAIN_MODULES:
        raise ValueError(
            f"validation.benchmark must be one of {sorted(_TRAIN_MODULES)}, "
            f"got {benchmark!r}"
        )
    return benchmark


def _import_split_api():
    """Keep the harness compatible with the split module's final package name."""
    try:
        module = importlib.import_module("val_anneal.split_datasets")
    except ModuleNotFoundError as error:
        if error.name != "val_anneal.split_datasets":
            raise
        module = importlib.import_module("val_anneal.datasets")
    return module.build_split_bundle


def _prepare_config(config: Any, benchmark: str, train_module: Any) -> None:
    if str(config.mode) != "train":
        raise ValueError("the offline annealing harness only supports mode=train")
    if hasattr(train_module, "apply_image_shape_override"):
        train_module.apply_image_shape_override(config)

    # Match Robomimic's existing pre-training configuration without constructing
    # the environment that its main() uses solely to discover this value.
    if config.task.obs_type == "image":
        config.task.obs_dim = config.network.emb_dim
    if benchmark == "robomimic" and config.network.network_type == "chiunet":
        old_horizon = int(config.task.horizon)
        config.task.horizon = int(2 ** np.ceil(np.log2(old_horizon)))
        if config.task.horizon != old_horizon:
            loguru.logger.warning(
                "ChiUNet requires a power-of-two horizon; changed {} to {}",
                old_horizon,
                config.task.horizon,
            )

    # Existing train() functions enter rollout evaluation solely via this
    # modulo.  A frequency beyond the final update makes that branch unreachable;
    # passing envs=None below then acts as an additional fail-fast guard.
    config.log.eval_freq = int(config.optimization.gradient_steps) + 1
    config.log.eval_episodes = 0
    config.log.save_video = False
    config.optimization.auto_resume = False


def _load_resume_state(config: Any, agent: TrainingAgent) -> dict | None:
    model_path = getattr(config.optimization, "model_path", None)
    if model_path is None or str(model_path) == "None":
        return None
    loguru.logger.info("Loading explicit resume checkpoint from {}", model_path)
    return agent.load(str(model_path), load_optimizer=True)


def _split_payload(bundle: Any) -> dict[str, Any]:
    split = bundle.split
    return {
        "seed": int(split.seed),
        "val_fraction": float(split.val_fraction),
        "train_ids": list(split.train_ids),
        "val_ids": list(split.val_ids),
        "all_ids": list(split.all_ids),
        "train_trajectories": int(bundle.train_count),
        "val_trajectories": int(bundle.val_count),
        "train_windows": int(bundle.train_sequence_count),
        "val_windows": int(bundle.val_sequence_count),
        "manifest_path": str(bundle.manifest_path),
    }


def run(config: Any) -> dict[str, Any]:
    """Run one configured training job and return its final validation report."""
    benchmark = _infer_benchmark(config)
    train_module = importlib.import_module(_TRAIN_MODULES[benchmark])
    _prepare_config(config, benchmark, train_module)
    if benchmark in {"libero", "robomimic"}:
        limit_threads(1)
    set_seed(
        int(config.optimization.seed),
        deterministic=bool(getattr(config.optimization, "deterministic", False)),
    )
    prepare_glitr(config)

    log_dir = Path(str(config.log.log_dir)).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    val_fraction = float(_validation_setting(config, "fraction", 0.2))
    split_seed = int(_validation_setting(config, "split_seed", 10000))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("validation.fraction must be strictly between 0 and 1")

    build_split_bundle = _import_split_api()
    bundle = build_split_bundle(
        config.task,
        benchmark,
        val_fraction=val_fraction,
        split_seed=split_seed,
        manifest_path=log_dir / "split_manifest.json",
    )
    logger: Logger | None = None
    try:
        if bundle.train_dataset.normalizer is not bundle.normalizer:
            raise RuntimeError(
                "training dataset does not use the split bundle normalizer"
            )
        if bundle.val_dataset.normalizer is not bundle.normalizer:
            raise RuntimeError(
                "validation dataset must reuse the exact training-set normalizer"
            )

        logger = Logger(config)
        agent = make_agent(config, TrainingAgent)
        record_run_metadata(config, bundle)
        resume_state = _load_resume_state(config, agent)
        loguru.logger.info(
            "Offline split ready: {} train trajectories / {} validation "
            "trajectories; no simulator will be constructed",
            bundle.train_count,
            bundle.val_count,
        )
        train_module.train(
            config,
            None,
            bundle.train_dataset,
            agent,
            logger,
            resume_state=resume_state,
        )
        logger.save_agent(agent=agent, identifier="final")

        if hasattr(train_module, "set_resource_phase"):
            train_module.set_resource_phase("offline_validation")
        metrics = validate_offline(
            config,
            bundle.val_dataset,
            agent,
            normalizer=bundle.normalizer,
        )
        metrics.update(
            {
                "train/num_trajectories": int(bundle.train_count),
                "train/num_windows": int(bundle.train_sequence_count),
                "val/num_trajectories": int(bundle.val_count),
            }
        )
        metrics.update(get_job_flop_metrics(agent))
        logger.log(
            {"step": int(config.optimization.gradient_steps) - 1, **metrics},
            category="eval",
        )

        report = {
            "schema_version": 1,
            "protocol": "trajectory_train_validation",
            "benchmark": benchmark,
            "rollout_evaluation": False,
            "checkpoint": str(log_dir / "models" / "model_final.pt"),
            "split": _split_payload(bundle),
            "metrics": metrics,
        }
        report_path = write_validation_report(log_dir / "validation.json", report)
        report_job_flops(agent)
        loguru.logger.info("Saved final offline validation report to {}", report_path)
        return report
    finally:
        bundle.close()
        wandb_client = getattr(logger, "_wandb", None) if logger is not None else None
        if wandb_client is not None:
            wandb_client.finish()


@hydra.main(version_base=None, config_path="../examples/configs", config_name="main")
def main(config: Any) -> None:
    run(config)


if __name__ == "__main__":
    main()
