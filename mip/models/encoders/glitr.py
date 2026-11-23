"""Original GLiTR architecture with a continuous robot-action output boundary.

The spatial, temporal, location and glimpse code is loaded from the unchanged
upstream checkout. Only the output width and supervision adapt classification
to continuous actions. No repository policy network or state fusion is used.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OFFICIAL_GLITR_INPUT_SIZE = 224
OFFICIAL_GLITR_GRID_SIZE = 14
VIT_PATCH_SIZE = 16


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_official_glitr_model():
    module_name = "glitr._official_glitr_model"
    if module_name in sys.modules:
        return sys.modules[module_name]

    model_path = _repo_root() / "third_party" / "GliTr" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing upstream GLiTR checkout at third_party/GliTr/model.py. "
            "Clone https://github.com/facebookresearch/GliTr.git there before training."
        )

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream GLiTR model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _install_timm_compat(module)
    return module


def _install_timm_compat(official) -> None:
    """Keep upstream GLiTR source untouched while running on modern timm."""
    if getattr(official, "_ROBOMIMIC_TIMM_COMPAT", False):
        return

    original_vit = official.VisionTransformer
    vit_kwargs = set(inspect.signature(original_vit.__init__).parameters) - {"self"}

    def filter_vit_kwargs(kwargs: dict) -> dict:
        return {key: value for key, value in kwargs.items() if key in vit_kwargs}

    class CompatVisionTransformer(original_vit):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **filter_vit_kwargs(kwargs))
            if not hasattr(self, "pre_logits"):
                self.pre_logits = nn.Identity()

    class CompatTemporalPatchEmbed(nn.Module):
        def __init__(self, img_size, patch_size, in_chans, embed_dim, **kwargs):
            super().__init__()
            del kwargs
            self.num_patches = img_size
            self.img_size = img_size
            self.patch_size = patch_size
            self.grid_size = (img_size,)
            if in_chans == embed_dim:
                self.embed_layer = nn.Identity()
            else:
                self.embed_layer = nn.Sequential(
                    nn.Linear(in_chans, embed_dim),
                    nn.LayerNorm(embed_dim, eps=1e-6),
                )

        def forward(self, x):
            return self.embed_layer(x)

    def create_model_compat(backbone, pretrained=False, **kwargs):
        ctor = getattr(official, backbone, None)
        if ctor is not None and backbone in getattr(official, "__all__", []):
            return ctor(pretrained=pretrained, **filter_vit_kwargs(kwargs))
        from timm.models import create_model as timm_create_model

        return timm_create_model(backbone, pretrained=pretrained, **kwargs)

    def causalforward_compat(self, x, *args, **kwargs):
        del args, kwargs
        batch, length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, length, channels)
        x = self.proj(x)
        return self.proj_drop(x)

    official.VisionTransformer = CompatVisionTransformer
    official.TemporalPatchEmbed = CompatTemporalPatchEmbed
    official.create_model = create_model_compat
    official.causalforward = causalforward_compat
    official._ROBOMIMIC_TIMM_COMPAT = True


def action_consistency_loss(student_outputs, teacher_outputs):
    """Continuous-output counterpart of upstream's categorical temporal KL."""
    return F.mse_loss(student_outputs, teacher_outputs.detach())


class ActionCriterion(nn.Module):
    """Supply real action targets despite upstream's class-label preprocessing.

    Both upstream forwards call this on flattened B*T predictions. The student
    converts labels to class indices first; binding continuous targets here
    avoids changing either forward method. No controls are invented for training.
    """

    def __init__(self):
        super().__init__()
        self.targets = None

    def forward(self, predictions, _class_labels):
        if self.targets is None:  # Inference has no supervision.
            return predictions.new_zeros(predictions.shape[0])
        return (predictions - self.targets).square().mean(dim=-1)


@dataclass
class GLiTROutput:
    actions: torch.Tensor
    action_loss: torch.Tensor
    spatial_loss: torch.Tensor
    temporal_loss: torch.Tensor
    masks: torch.Tensor


class GLiTRPolicy(nn.Module):
    """One original VideoTransformer, with its head sized for robot actions."""

    def __init__(self, task, glitr, teacher=None):
        super().__init__()
        self.camera_key = glitr.camera_key
        self.horizon = int(task.horizon)
        self.act_dim = int(task.act_dim)
        self.obs_steps = int(task.obs_steps)
        self.output_dim = self.horizon * self.act_dim
        self.input_size = OFFICIAL_GLITR_INPUT_SIZE
        if glitr.input_size != self.input_size:
            raise ValueError("GLiTR requires input_size=224")
        if not 1 <= glitr.num_patches_in_glimpse <= OFFICIAL_GLITR_GRID_SIZE:
            raise ValueError("GLiTR requires 1-14 glimpse patches")
        checkpoint = Path(glitr.pretrained_dir) / "ibot_vits_16_checkpoint_teacher.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing required upstream iBOT checkpoint: {checkpoint}"
            )
        official = _load_official_glitr_model()
        self.video_transformer = official.VideoTransformer(
            backbone=glitr.backbone,
            num_classes=self.output_dim,
            num_frames_per_video=self.obs_steps,
            drop=glitr.drop,
            drop_path=glitr.drop_path,
            num_patches_in_glimpse=glitr.num_patches_in_glimpse,
            criterion=ActionCriterion(),
            attntype="teacher" if teacher is None else "student",
            pretrained_dir=glitr.pretrained_dir,
            teacher=teacher.video_transformer if teacher is not None else None,
        )
        # Keep upstream's consistency locations and gradient stops; compare
        # continuous controls with MSE instead of class-probability KL.
        self.video_transformer.kld_loss = action_consistency_loss
        self.register_buffer("imagenet_mean", IMAGENET_MEAN.clone())
        self.register_buffer("imagenet_std", IMAGENET_STD.clone())

    def _prepare_image_sequence(self, image_seq):
        batch, steps, channels, height, width = image_seq.shape
        if steps != self.obs_steps or channels != 3:
            raise ValueError(f"GLiTR expects B,{self.obs_steps},3,H,W images")
        image = (image_seq.reshape(batch * steps, channels, height, width) + 1.0) / 2.0
        if (height, width) != (self.input_size, self.input_size):
            image = F.interpolate(
                image,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        image = (image - self.imagenet_mean.to(image.dtype)) / self.imagenet_std.to(
            image.dtype
        )
        return image.reshape(batch, steps, channels, self.input_size, self.input_size)

    def forward(self, obs, actions=None):
        images = self._prepare_image_sequence(obs[self.camera_key])
        batch, steps = images.shape[:2]
        criterion = self.video_transformer.criterion
        if actions is not None:
            if actions.shape != (batch, self.horizon, self.act_dim):
                raise ValueError(
                    f"Expected actions shaped {(batch, self.horizon, self.act_dim)}"
                )
            labels = actions.flatten(1)
            criterion.targets = (
                labels[:, None]
                .expand(-1, steps, -1)
                .reshape(batch * steps, self.output_dim)
            )
        else:
            # Upstream requires labels even at inference; these unused placeholders
            # contribute no supervision and do not affect the prediction.
            labels = images.new_zeros(batch, 1)
        try:
            predictions, action_loss, _, spatial_loss, temporal_loss, masks = (
                self.video_transformer(images, labels)
            )
        finally:
            criterion.targets = None
        return GLiTROutput(
            actions=predictions[-1].reshape(batch, self.horizon, self.act_dim),
            action_loss=action_loss,
            spatial_loss=spatial_loss,
            temporal_loss=temporal_loss,
            masks=masks,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.video_transformer.teacher is not None:
            self.video_transformer.teacher.eval()
        return self
