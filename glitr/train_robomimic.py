"""Train/evaluate the upstream GLiTR Robomimic adapter.

This entrypoint intentionally does not modify or depend on the ``mip`` encoder
registry. It uses the existing task/network/optimization configs as library
inputs, and runs ``GLiTRAgent`` with its original temporal action-output head.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import hydra
import loguru
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from examples.train_robomimic import (  # noqa: E402
    apply_image_shape_override,
    checkpoint_resolution_suffix,
    eval,
    get_eval_step_list,
)
from glitr.agent import GLiTRAgent  # noqa: E402
from glitr.config import GLiTRConfig  # noqa: E402
from mip.config import Config  # noqa: E402
from mip.data.datasets.robomimic_dataset import make_dataset  # noqa: E402
from mip.envs.robomimic.robomimic_env import make_vec_env  # noqa: E402
from mip.flop_accounting import (  # noqa: E402
    get_job_flop_metrics,
    report_job_flops,
)
from mip.glitr import checkpoint_name, export_checkpoint, prepare_glitr  # noqa: E402
from mip.logger import (  # noqa: E402
    Logger,
    compute_average_metrics,
    update_best_metrics,
)
from mip.reproducibility import (  # noqa: E402
    dataloader_seed_kwargs,
    record_run_metadata,
)
from mip.scheduler import WarmupAnnealingScheduler  # noqa: E402
from mip.torch_utils import limit_threads, set_seed  # noqa: E402

torch.set_float32_matmul_precision("high")


@contextmanager
def timed(section: str, record_dict: dict):
    start = time.perf_counter()
    yield
    record_dict[section].append(time.perf_counter() - start)


def ensure_glitr_config(config) -> None:
    if hasattr(config, "glitr"):
        defaults = OmegaConf.structured(GLiTRConfig())
        merged = OmegaConf.merge(defaults, config.glitr)
        with open_dict(config):
            config.glitr = merged
        return
    with open_dict(config):
        config.glitr = OmegaConf.structured(GLiTRConfig())


def validate_official_glitr_config(config) -> None:
    image_shape = getattr(config.task, "image_shape", None)
    if config.task.obs_type != "image" or image_shape is None:
        return
    height, width = int(image_shape[0]), int(image_shape[1])
    if height != width:
        raise ValueError(
            f"GLiTR expects square Robomimic image datasets, got {image_shape}"
        )
    if int(config.glitr.input_size) != GLiTRConfig().input_size:
        raise ValueError(
            "The upstream GLiTR adapter is fixed to glitr.input_size=224. "
            f"Got {config.glitr.input_size}."
        )
    if int(config.glitr.num_patches_in_glimpse) > 14:
        raise ValueError(
            "num_patches_in_glimpse must be <= 14 for upstream GLiTR's 14x14 grid."
        )


def train(config: Config, envs, dataset, agent: GLiTRAgent, logger, resume_state=None):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=8 if config.task.obs_type == "image" else 4,
        shuffle=True,
        **dataloader_seed_kwargs(config),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    def loop(loader):
        while True:
            yield from loader

    loop_loader = loop(dataloader)
    lr_scheduler = CosineAnnealingLR(
        agent.optimizer,
        T_max=config.optimization.gradient_steps,
    )
    warmup_scheduler = WarmupAnnealingScheduler(
        max_steps=config.optimization.gradient_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        rampup_ratio=config.optimization.rampup_ratio,
        min_value=config.optimization.min_value,
        max_value=config.optimization.max_value,
    )

    start_step = 0
    best_metrics = {}
    eval_history = []
    if resume_state:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        for _ in range(start_step):
            lr_scheduler.step()

    info_list = []
    start_time = time.time()
    perf_times = {"data_load": [], "preprocess": [], "update": [], "total_step": []}
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        with timed("total_step", perf_times):
            with timed("data_load", perf_times):
                batch = next(loop_loader)

            with timed("preprocess", perf_times):
                if config.task.obs_type != "image":
                    raise ValueError(
                        "GLiTR Robomimic entrypoint currently expects image tasks"
                    )
                obs_batch = batch["obs"]
                obs = {}
                for key in obs_batch:
                    obs[key] = obs_batch[key][:, : config.task.obs_steps, :].to(
                        config.optimization.device
                    )
                act = batch["action"].to(config.optimization.device)
                act = act[:, : config.task.horizon, :]

            with timed("update", perf_times):
                delta_t_scalar = warmup_scheduler(n_gradient_step)
                delta_t = torch.full(
                    (act.shape[0],),
                    delta_t_scalar,
                    device=config.optimization.device,
                )
                info = agent.update(act, obs, delta_t)
                lr_scheduler.step()

            info = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in info.items()
            }
            info_list.append(info)

        if ((n_gradient_step + 1) % config.log.log_freq) == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info_list[-1]:
                try:
                    metrics[key] = np.nanmean([info[key] for info in info_list])
                except Exception:
                    metrics[key] = np.nan
            if perf_times["total_step"]:
                metrics["perf/data_load_ms"] = (
                    np.mean(perf_times["data_load"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/preprocess_ms"] = (
                    np.mean(perf_times["preprocess"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/update_ms"] = (
                    np.mean(perf_times["update"][-config.log.log_freq :]) * 1000
                )
                metrics["perf/total_step_ms"] = (
                    np.mean(perf_times["total_step"][-config.log.log_freq :]) * 1000
                )
            metrics.update(get_job_flop_metrics(agent))
            logger.log(metrics, category="train")
            info_list = []

        if ((n_gradient_step + 1) % config.log.save_freq) == 0:
            logger.save_agent(agent=agent, identifier="latest")

        if ((n_gradient_step + 1) % config.log.eval_freq) == 0:
            agent.eval()
            metrics = {"step": n_gradient_step}
            num_steps_list = get_eval_step_list(config)
            for num_steps in num_steps_list:
                metrics.update(eval(config, envs, dataset, agent, logger, num_steps))

            old_best_metrics = best_metrics.copy()
            best_metrics = update_best_metrics(best_metrics, metrics)
            eval_history.append(metrics.copy())
            avg_metrics = compute_average_metrics(eval_history)
            primary_metric_key = f"mean_success_{num_steps_list[0]}"
            if primary_metric_key in metrics:
                primary_metric = metrics[primary_metric_key]
                is_new_best = np.isfinite(primary_metric) and (
                    primary_metric_key not in old_best_metrics
                    or primary_metric > old_best_metrics[primary_metric_key]
                )
                if is_new_best:
                    success_rate = primary_metric
                    loguru.logger.info(
                        f"New best model! {primary_metric_key} = {success_rate:.4f}"
                    )
                    training_state = {
                        "n_gradient_step": n_gradient_step,
                        "best_metrics": best_metrics,
                        "eval_history": eval_history,
                    }
                    logger.save_agent(agent=agent, identifier="best")
                    glimpse_suffix = (
                        ""
                        if config.glitr.stage == "teacher"
                        else f"_p{config.glitr.num_patches_in_glimpse}"
                    )
                    checkpoint_base_name = (
                        f"glitr_{config.glitr.stage}_{config.task.env_name}_"
                        f"{config.task.env_type}_{config.task.obs_type}_"
                        f"{config.optimization.loss_type}_{config.network.network_type}_"
                        f"{config.network.emb_dim}_seed{config.optimization.seed}"
                        f"{glimpse_suffix}"
                        f"{checkpoint_resolution_suffix(config)}"
                    )
                    logger.save_global_checkpoint(
                        agent,
                        checkpoint_name(config, checkpoint_base_name),
                        success_rate,
                        training_state=training_state,
                    )

            for key, value in best_metrics.items():
                metrics[f"best_{key}"] = value
            for key, value in avg_metrics.items():
                metrics[key] = value
            metrics.update(get_job_flop_metrics(agent))
            logger.log(metrics, category="eval")
            agent.train()

    report_job_flops(agent)


@hydra.main(
    version_base=None, config_path="../examples/configs", config_name="glitr_robomimic"
)
def main(config):
    ensure_glitr_config(config)
    if str(config.optimization.device).startswith("cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    set_seed(
        config.optimization.seed,
        deterministic=getattr(config.optimization, "deterministic", False),
    )
    limit_threads(1)
    apply_image_shape_override(config)
    validate_official_glitr_config(config)

    prepare_glitr(config)

    logger = Logger(config)
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    del obs
    config.task.obs_dim = config.network.emb_dim
    dataset = make_dataset(config.task)

    agent = GLiTRAgent(config)
    record_run_metadata(config, dataset)
    resume_state = None
    if config.optimization.model_path and config.optimization.model_path != "None":
        resume_state = agent.load(config.optimization.model_path, load_optimizer=True)

    if config.mode == "train":
        train(config, envs, dataset, agent, logger, resume_state=resume_state)
        export_checkpoint(config, agent)
    elif config.mode == "eval":
        agent.eval()
        for num_steps in get_eval_step_list(config):
            metrics = {"step": num_steps}
            metrics.update(eval(config, envs, dataset, agent, logger, num_steps))
            metrics.update(get_job_flop_metrics(agent))
            logger.log(metrics, category="eval")
            for key, val in metrics.items():
                if "mean_success" in key:
                    loguru.logger.info(f"{key} - {val}")
    else:
        raise ValueError(f"Illegal mode: {config.mode}")


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
