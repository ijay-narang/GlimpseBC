"""Single-task MIP training and evaluation pipeline for LIBERO."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Configure headless rendering before importing LIBERO / MuJoCo modules.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import loguru
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mip.agent import TrainingAgent
from mip.config import Config
from mip.data.datasets.libero_dataset import make_dataset
from mip.data.loader_utils import loop_dataloader
from mip.envs.libero import make_vec_env
from mip.flop_accounting import (
    get_job_flop_metrics,
    report_job_flops,
    track_evaluation_flops,
)
from mip.glitr import checkpoint_name, export_checkpoint, make_agent, prepare_glitr
from mip.logger import Logger, compute_average_metrics, update_best_metrics
from mip.reproducibility import (
    dataloader_seed_kwargs,
    evaluation_seed,
    record_run_metadata,
    reproducible_evaluation,
)
from mip.samplers import get_default_step_list
from mip.scheduler import WarmupAnnealingScheduler
from mip.torch_utils import limit_threads, set_seed

torch.set_float32_matmul_precision("high")


def get_eval_step_list(config: Config) -> list[int]:
    eval_num_steps = getattr(config.optimization, "eval_num_steps", None)
    if eval_num_steps is not None:
        return [int(value) for value in eval_num_steps]
    return get_default_step_list(config.optimization.loss_type)


def checkpoint_base_name(config: Config) -> str:
    suite = str(config.task.libero_suite)
    task_id = int(config.task.libero_task_id)
    return checkpoint_name(
        config,
        (
            f"libero_{suite}_task{task_id}_"
            f"{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        ),
    )


def _prepare_obs(config: Config, dataset, raw_obs) -> dict[str, torch.Tensor]:
    if config.task.obs_type != "image":
        raise ValueError(
            "The modular LIBERO pipeline currently supports image observations"
        )
    obs = {}
    for key, value in raw_obs.items():
        value = value.astype(np.float32)
        value = dataset.normalizer["obs"][key].normalize(value)
        obs[key] = torch.as_tensor(
            value,
            device=config.optimization.device,
            dtype=torch.float32,
        )
    return obs


def train(config: Config, envs, dataset, agent, logger, resume_state=None):
    """Train on one LIBERO task using the existing MIP agent contract."""
    num_workers = int(getattr(config.task, "num_workers", 4))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        num_workers=num_workers,
        shuffle=True,
        **dataloader_seed_kwargs(config),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    loop_loader = loop_dataloader(dataloader)
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
    if resume_state is not None:
        start_step = resume_state.get("n_gradient_step", 0) + 1
        best_metrics = resume_state.get("best_metrics", {})
        eval_history = resume_state.get("eval_history", [])
        for _ in range(start_step):
            lr_scheduler.step()
        loguru.logger.info(f"Resuming LIBERO training from step {start_step}")

    info_list = []
    start_time = time.time()
    for n_gradient_step in range(start_step, config.optimization.gradient_steps):
        batch = next(loop_loader)
        obs = {
            key: value[:, : config.task.obs_steps].to(config.optimization.device)
            for key, value in batch["obs"].items()
        }
        action = batch["action"][:, : config.task.horizon].to(
            config.optimization.device
        )
        delta_t_scalar = warmup_scheduler(n_gradient_step)
        delta_t = torch.full(
            (action.shape[0],),
            delta_t_scalar,
            device=config.optimization.device,
        )
        info = agent.update(action, obs, delta_t)
        info_list.append(
            {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in info.items()
            }
        )
        lr_scheduler.step()

        if (n_gradient_step + 1) % config.log.log_freq == 0:
            metrics = {
                "step": n_gradient_step,
                "total_time": time.time() - start_time,
                "lr": lr_scheduler.get_last_lr()[0],
                "delta_t": delta_t_scalar,
            }
            for key in info_list[-1]:
                metrics[key] = np.nanmean(
                    [item[key] for item in info_list if key in item]
                )
            metrics.update(get_job_flop_metrics(agent))
            logger.log(metrics, category="train")
            info_list = []

        if (n_gradient_step + 1) % config.log.save_freq == 0:
            logger.save_agent(agent=agent, identifier="latest")

        if (n_gradient_step + 1) % config.log.eval_freq != 0:
            continue

        agent.eval()
        metrics = {"step": n_gradient_step}
        num_steps_list = get_eval_step_list(config)
        for num_steps in num_steps_list:
            metrics.update(evaluate(config, envs, dataset, agent, logger, num_steps))

        old_best_metrics = best_metrics.copy()
        best_metrics = update_best_metrics(best_metrics, metrics)
        eval_history.append(metrics.copy())
        avg_metrics = compute_average_metrics(eval_history)
        primary_key = f"mean_success_{num_steps_list[0]}"
        if primary_key in metrics and (
            primary_key not in old_best_metrics
            or metrics[primary_key] > old_best_metrics[primary_key]
        ):
            logger.save_agent(agent=agent, identifier="best")
            logger.save_global_checkpoint(
                agent,
                checkpoint_base_name(config),
                metrics[primary_key],
                training_state={
                    "n_gradient_step": n_gradient_step,
                    "best_metrics": best_metrics,
                    "eval_history": eval_history,
                },
            )
        for key, value in best_metrics.items():
            metrics[f"best_{key}"] = value
        metrics.update(avg_metrics)
        metrics.update(get_job_flop_metrics(agent))
        logger.log(metrics, category="eval")
        agent.train()

    report_job_flops(agent)


@torch.no_grad()
@track_evaluation_flops("libero")
@reproducible_evaluation
def evaluate(config: Config, envs, dataset, agent, logger, num_steps=1):
    """Evaluate against LIBERO's fixed initial states and sparse success signal."""
    del logger
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_success: list[float] = []
    completed = 0
    batch_index = 0

    while completed < int(config.log.eval_episodes):
        for env_index, env in enumerate(envs.envs):
            env.seed(
                (
                    evaluation_seed(config)
                    + batch_index * int(config.task.num_envs)
                    + env_index
                )
                % 2**32
            )
        raw_obs, _ = envs.reset()
        batch_size = min(
            int(config.task.num_envs), int(config.log.eval_episodes) - completed
        )
        rewards = np.zeros(int(config.task.num_envs), dtype=np.float32)
        successes = np.zeros(int(config.task.num_envs), dtype=bool)
        done = np.zeros(int(config.task.num_envs), dtype=bool)
        steps = 0

        while steps < int(config.task.max_episode_steps) and not np.all(done):
            obs = _prepare_obs(config, dataset, raw_obs)
            act_0 = torch.randn(
                (
                    int(config.task.num_envs),
                    int(config.task.horizon),
                    int(config.task.act_dim),
                ),
                device=config.optimization.device,
            )
            action = agent.sample(
                act_0=act_0,
                obs=obs,
                num_steps=num_steps,
                use_ema=True,
            )
            action = dataset.normalizer["action"].unnormalize(
                action.detach().cpu().numpy()
            )
            start = int(config.task.obs_steps) - 1
            action = action[:, start : start + int(config.task.act_steps)]
            raw_obs, reward, terminated, truncated, _ = envs.step(action)
            reward = np.asarray(reward, dtype=np.float32)
            rewards += np.where(done, 0.0, reward)
            successes |= reward > 0
            done |= np.asarray(terminated) | np.asarray(truncated)
            steps += int(config.task.act_steps)

        episode_rewards.extend(rewards[:batch_size].tolist())
        episode_steps.extend(
            [min(steps, int(config.task.max_episode_steps))] * batch_size
        )
        episode_success.extend(successes[:batch_size].astype(np.float32).tolist())
        completed += batch_size
        batch_index += 1

    metrics = {
        f"mean_step_{num_steps}": float(np.mean(episode_steps)),
        f"mean_reward_{num_steps}": float(np.mean(episode_rewards)),
        f"mean_success_{num_steps}": float(np.mean(episode_success)),
    }
    loguru.logger.info(
        f"LIBERO NFE={num_steps} mean_step={metrics[f'mean_step_{num_steps}']:.2f} "
        f"mean_reward={metrics[f'mean_reward_{num_steps}']:.4f} "
        f"mean_success={metrics[f'mean_success_{num_steps}']:.4f}"
    )
    return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    """Run the standalone LIBERO pipeline."""
    limit_threads(1)
    set_seed(
        config.optimization.seed,
        deterministic=getattr(config.optimization, "deterministic", False),
    )
    prepare_glitr(config)
    logger = Logger(config)
    dataset = make_dataset(config.task)
    envs = make_vec_env(config.task, seed=config.optimization.seed)
    agent = make_agent(config, TrainingAgent)
    record_run_metadata(config, dataset)
    resume_state = None

    try:
        if config.optimization.model_path and config.optimization.model_path != "None":
            resume_state = agent.load(
                config.optimization.model_path,
                load_optimizer=config.mode == "train",
            )
        elif config.mode == "train" and config.optimization.auto_resume:
            checkpoint_path = logger.find_latest_checkpoint(
                checkpoint_base_name(config)
            )
            if checkpoint_path:
                resume_state = agent.load(str(checkpoint_path), load_optimizer=True)

        if config.mode == "train":
            train(config, envs, dataset, agent, logger, resume_state=resume_state)
            export_checkpoint(config, agent)
        elif config.mode == "eval":
            agent.eval()
            for num_steps in get_eval_step_list(config):
                metrics = {"step": num_steps}
                metrics.update(
                    evaluate(config, envs, dataset, agent, logger, num_steps)
                )
                metrics.update(get_job_flop_metrics(agent))
                logger.log(metrics, category="eval")
        else:
            raise ValueError(f"Illegal mode {config.mode!r}")
    finally:
        dataset.close()
        envs.close()


if __name__ == "__main__":
    main()
