"""Connect GLiTR's own robot-action model to benchmark runner infrastructure."""

import os
import shutil
from pathlib import Path

import loguru
from omegaconf import OmegaConf, open_dict

from glitr.config import GLiTRConfig


def uses_glitr(config):
    return getattr(config, "glitr", None) is not None


def _stage_tag(config):
    stage = "student" if config.glitr.stage == "eval" else config.glitr.stage
    if stage == "teacher":
        return f"glitr_actions_teacher_{config.glitr.camera_key}"
    return f"glitr_actions_{stage}_p{config.glitr.num_patches_in_glimpse}_{config.glitr.camera_key}"


def prepare_glitr(config):
    """Validate the optional GLiTR configuration before loading benchmark data."""
    if not uses_glitr(config):
        return
    with open_dict(config):
        config.glitr = OmegaConf.merge(
            OmegaConf.structured(GLiTRConfig()), config.glitr
        )
    glitr = config.glitr
    if config.task.obs_type != "image":
        raise ValueError("Upstream GLiTR requires an image task")
    if config.optimization.loss_type != "regression":
        raise ValueError(
            "GLiTR robot actions require optimization.loss_type=regression"
        )
    if config.network.network_type != "glitr":
        raise ValueError("GLiTR uses its own temporal head; select network=glitr")
    if glitr.stage not in {"teacher", "student", "eval"}:
        raise ValueError(f"Unknown GLiTR stage: {glitr.stage}")
    if glitr.stage == "eval" and config.mode != "eval":
        raise ValueError("glitr.stage=eval requires mode=eval")
    if glitr.stage != "teacher" and not glitr.teacher_checkpoint:
        raise ValueError("GLiTR student/eval requires glitr.teacher_checkpoint")
    if glitr.input_size != 224 or not 1 <= glitr.num_patches_in_glimpse <= 14:
        raise ValueError(
            "Upstream GLiTR requires input_size=224 and 1-14 glimpse patches"
        )
    if config.optimization.use_compile or config.optimization.use_cudagraphs:
        raise ValueError("GLiTR uses eager execution; disable compile and CUDA graphs")
    if config.task.horizon < config.task.obs_steps - 1 + config.task.act_steps:
        raise ValueError(
            "task.horizon must cover observation history and executed actions"
        )
    rgb_keys = sorted(
        key
        for key, attr in config.task.shape_meta["obs"].items()
        if attr.get("type") == "rgb"
    )
    if not glitr.camera_key:
        glitr.camera_key = next(
            (
                key
                for key in ("agentview_image", "agentview_rgb", "image")
                if key in rgb_keys
            ),
            rgb_keys[0] if rgb_keys else None,
        )
    if glitr.camera_key not in rgb_keys:
        raise ValueError(f"glitr.camera_key must select an RGB observation: {rgb_keys}")
    if not glitr.pretrained_dir:
        glitr.pretrained_dir = os.environ.get(
            "GLITR_PRETRAINED_DIR",
            str(Path(__file__).resolve().parents[1] / "glitr/pretrained"),
        )
    # The agent supplies the entire model; ordinary encoder/policy factories are unused.
    config.network.encoder_type = "glitr"
    tag = _stage_tag(config)
    if Path(config.log.log_dir).name != tag:
        config.log.log_dir = str(Path(config.log.log_dir) / tag)
    if not config.log.exp_name.startswith(tag + "_"):
        config.log.exp_name = f"{tag}_{config.log.exp_name}"


def make_agent(config, default_factory):
    """Keep the benchmark's normal agent unless the GLiTR config is selected."""
    if not uses_glitr(config):
        return default_factory(config)
    from glitr.agent import GLiTRAgent

    return GLiTRAgent(config)


def checkpoint_name(config, base_name):
    """Separate the shared teacher and crop-specific students from other policies."""
    if not uses_glitr(config):
        return base_name
    return f"{_stage_tag(config)}_{base_name}"


def export_checkpoint(config, agent):
    """Export the best saved GLiTR policy, or the final policy for a short run."""
    if not uses_glitr(config) or not config.glitr.export_checkpoint:
        return
    destination = Path(config.glitr.export_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best = Path(config.log.log_dir) / "models/model_best.pt"
    if best.is_file():
        if best.resolve() != destination.resolve():
            shutil.copy2(best, destination)
    else:
        agent.save(str(destination))
    loguru.logger.info(f"Exported GLiTR {config.glitr.stage} checkpoint: {destination}")
