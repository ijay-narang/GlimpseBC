"""Vanilla saccade encoder for image observations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from mip.models.encoders.base import BaseEncoder
from mip.models.layers import replace_submodules


def _replace_bn_with_group_norm(model: nn.Module) -> nn.Module:
    return replace_submodules(
        root_module=model,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // 16),
            num_channels=x.num_features,
        ),
    )


class ResNetPatchEncoder(nn.Module):
    """Encode cropped glimpses with a ResNet backbone."""

    def __init__(
        self,
        model_name: str = "resnet18",
        emb_dim: int = 256,
        use_group_norm: bool = False,
    ):
        super().__init__()
        if not model_name.startswith("resnet"):
            raise ValueError(
                "VanillaSaccadeImageEncoder only supports resnet backbones, "
                f"got {model_name}"
            )
        model_fn = getattr(torchvision.models, model_name)
        backbone = model_fn(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        if use_group_norm:
            backbone = _replace_bn_with_group_norm(backbone)
        self.backbone = backbone
        self.proj = nn.Linear(feature_dim, emb_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(patch))


def soft_crop_patches(
    images: torch.Tensor,
    locations: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Differentiably crop square patches with normalized top-left locations."""
    batch_size, channels, image_h, image_w = images.shape
    patch_size = min(patch_size, image_h, image_w)
    max_y = image_h - patch_size
    max_x = image_w - patch_size

    top = locations[:, 0] * max_y
    left = locations[:, 1] * max_x
    y1 = top
    y2 = top + patch_size
    x1 = left
    x2 = left + patch_size

    theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = patch_size / image_w
    theta[:, 1, 1] = patch_size / image_h
    theta[:, 0, 2] = -1 + (x1 + x2) / image_w
    theta[:, 1, 2] = -1 + (y1 + y2) / image_h

    grid = F.affine_grid(
        theta,
        torch.Size((batch_size, channels, patch_size, patch_size)),
        align_corners=False,
    )
    return F.grid_sample(images, grid, align_corners=False)


def hard_crop_patches(
    images: torch.Tensor,
    locations: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Crop discrete square patches with rounded normalized top-left locations."""
    batch_size, channels, image_h, image_w = images.shape
    patch_size = min(patch_size, image_h, image_w)
    max_y = image_h - patch_size
    max_x = image_w - patch_size

    top = torch.round(locations[:, 0] * max_y).long().clamp(0, max_y)
    left = torch.round(locations[:, 1] * max_x).long().clamp(0, max_x)

    yy = torch.arange(patch_size, device=images.device)
    xx = torch.arange(patch_size, device=images.device)
    y_idx = top[:, None] + yy[None, :]
    x_idx = left[:, None] + xx[None, :]

    flat_images = images.reshape(batch_size, channels, image_h * image_w)
    flat_idx = y_idx[:, :, None] * image_w + x_idx[:, None, :]
    flat_idx = flat_idx.reshape(batch_size, 1, patch_size * patch_size)
    flat_idx = flat_idx.expand(batch_size, channels, patch_size * patch_size)
    patches = torch.gather(flat_images, dim=-1, index=flat_idx)
    return patches.reshape(batch_size, channels, patch_size, patch_size)


class VanillaSaccadeImageEncoder(BaseEncoder):
    """Glimpse-history encoder with soft train crops and hard eval crops.

    For each camera, the encoder crops a sequence of glimpses. The first crop
    starts at a learned initial location, and each next crop location is predicted
    from the temporal memory of previous glimpse tokens.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "resnet18",
        emb_dim: int = 256,
        patch_size: int | None = None,
        resize_shape: tuple[int, int] | None = None,
        crop_shape: tuple[int, int] | None = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        imagenet_norm: bool = False,
        use_seq: bool = True,
        keep_horizon_dims: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        del random_crop

        self.shape_meta = shape_meta
        self.emb_dim = emb_dim
        self.resize_shape = tuple(resize_shape) if resize_shape is not None else None
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.imagenet_norm = imagenet_norm
        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims

        obs_meta = shape_meta["obs"]
        self.rgb_keys = sorted(
            key
            for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "rgb"
        )
        self.low_dim_keys = sorted(
            key
            for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        )
        if not self.rgb_keys:
            raise ValueError("VanillaSaccadeImageEncoder requires at least one rgb obs key")

        first_rgb_shape = tuple(obs_meta[self.rgb_keys[0]]["shape"])
        if patch_size is None:
            if self.crop_shape is not None:
                patch_size = min(self.crop_shape)
            else:
                patch_size = min(48, first_rgb_shape[-2], first_rgb_shape[-1])
        self.patch_size = int(patch_size)

        self.patch_encoders = nn.ModuleDict(
            {
                key: ResNetPatchEncoder(
                    model_name=rgb_model_name,
                    emb_dim=emb_dim,
                    use_group_norm=use_group_norm,
                )
                for key in self.rgb_keys
            }
        )
        self.location_encoders = nn.ModuleDict(
            {key: nn.Linear(2, emb_dim) for key in self.rgb_keys}
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_models = nn.ModuleDict(
            {
                key: nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_transformer_layers,
                )
                for key in self.rgb_keys
            }
        )
        self.location_heads = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(emb_dim),
                    nn.Linear(emb_dim, emb_dim),
                    nn.GELU(),
                    nn.Linear(emb_dim, 2),
                    nn.Sigmoid(),
                )
                for key in self.rgb_keys
            }
        )
        self.initial_locations = nn.ParameterDict(
            {
                key: nn.Parameter(torch.full((2,), 0.5))
                for key in self.rgb_keys
            }
        )

        low_dim_total = sum(
            int(torch.tensor(obs_meta[key]["shape"]).prod())
            for key in self.low_dim_keys
        )
        self.low_dim_encoder = (
            nn.Sequential(
                nn.Linear(low_dim_total, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
            if low_dim_total > 0
            else None
        )
        final_input_dim = emb_dim * len(self.rgb_keys)
        if self.low_dim_encoder is not None:
            final_input_dim += emb_dim
        self.output_proj = nn.Sequential(
            nn.Linear(final_input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        if imagenet_norm:
            self.register_buffer(
                "imagenet_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "imagenet_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_shape is not None:
            image = F.interpolate(
                image,
                size=self.resize_shape,
                mode="bilinear",
                align_corners=False,
            )
        if self.imagenet_norm:
            image = (image - self.imagenet_mean.to(image.dtype)) / self.imagenet_std.to(
                image.dtype
            )
        return image

    def _crop(self, image: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        if self.training:
            return soft_crop_patches(image, location, self.patch_size)
        return hard_crop_patches(image, location, self.patch_size)

    def _encode_rgb_sequence(self, key: str, image_seq: torch.Tensor) -> torch.Tensor:
        if not self.use_seq:
            image_seq = image_seq.unsqueeze(1)

        batch_size, seq_len = image_seq.shape[:2]
        location = torch.sigmoid(self.initial_locations[key]).expand(batch_size, 2)
        tokens = []
        memory = None

        for step in range(seq_len):
            image = self._prepare_image(image_seq[:, step])
            patch = self._crop(image, location)
            patch_feature = self.patch_encoders[key](patch)
            location_feature = self.location_encoders[key](location)
            token = patch_feature + location_feature
            tokens.append(token)

            token_seq = torch.stack(tokens, dim=1)
            memory = self.temporal_models[key](token_seq)
            location = self.location_heads[key](memory[:, -1])

        if memory is None:
            raise ValueError("Empty image sequence passed to VanillaSaccadeImageEncoder")
        return memory if self.use_seq else memory[:, 0]

    def _encode_low_dim(self, obs_dict, batch_size: int, seq_len: int) -> torch.Tensor | None:
        if self.low_dim_encoder is None:
            return None
        low_dim = []
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if self.use_seq:
                data = data.reshape(batch_size, seq_len, -1)
            else:
                data = data.reshape(batch_size, 1, -1)
            low_dim.append(data)
        low_dim = torch.cat(low_dim, dim=-1)
        low_dim = low_dim.reshape(batch_size * seq_len, -1)
        low_dim = self.low_dim_encoder(low_dim)
        return low_dim.reshape(batch_size, seq_len, self.emb_dim)

    def forward(self, obs_dict, mask=None):
        del mask

        first_image = obs_dict[self.rgb_keys[0]]
        if self.use_seq:
            batch_size, seq_len = first_image.shape[:2]
        else:
            batch_size = first_image.shape[0]
            seq_len = 1

        features = []
        for key in self.rgb_keys:
            features.append(self._encode_rgb_sequence(key, obs_dict[key]))

        low_dim = self._encode_low_dim(obs_dict, batch_size, seq_len)
        if low_dim is not None:
            features.append(low_dim if self.use_seq else low_dim[:, 0])

        if self.use_seq:
            encoded = torch.cat(features, dim=-1)
            encoded = self.output_proj(encoded.reshape(batch_size * seq_len, -1))
            encoded = encoded.reshape(batch_size, seq_len, self.emb_dim)
            if not self.keep_horizon_dims:
                encoded = encoded.reshape(batch_size, seq_len * self.emb_dim)
            return encoded

        encoded = torch.cat(features, dim=-1)
        return self.output_proj(encoded)
