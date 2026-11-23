"""Shared settings for the minimal GLiTR robot-action adaptation."""

from dataclasses import dataclass


@dataclass
class GLiTRConfig:
    stage: str = "teacher"
    teacher_checkpoint: str | None = None
    export_checkpoint: str | None = None
    camera_key: str | None = None
    backbone: str = "deit_small_patch16_224"
    input_size: int = 224
    num_patches_in_glimpse: int = 3
    pretrained_dir: str | None = None
    drop: float = 0.0
    drop_path: float = 0.1
    spatial_loss_weight: float = 1.0
    temporal_loss_weight: float = 1.0
    base_lr_backbone: float = 1e-5
    base_lr_temphead: float | None = None
    base_lr_temploc: float | None = None
    weight_decay: float = 0.05
