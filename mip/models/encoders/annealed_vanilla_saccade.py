"""Vanilla saccade encoder with a training-time field-of-view curriculum."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mip.models.encoders.vanilla_saccade import VanillaSaccadeImageEncoder


def soft_crop_patches_with_fov(
    images: torch.Tensor,
    locations: torch.Tensor,
    source_crop_size: float,
    output_size: int,
) -> torch.Tensor:
    """Differentiably sample a source field of view at fixed output resolution.

    Locations use the vanilla encoder's normalized top-left convention. The
    source field of view may be wider than the tensor passed to the patch
    encoder, in which case ``grid_sample`` downsamples it to ``output_size``.
    """
    batch_size, channels, image_h, image_w = images.shape
    output_size = min(output_size, image_h, image_w)
    source_crop_size = float(
        max(output_size, min(source_crop_size, image_h, image_w))
    )

    top = locations[:, 0] * (image_h - source_crop_size)
    left = locations[:, 1] * (image_w - source_crop_size)

    theta = images.new_zeros(batch_size, 2, 3)
    theta[:, 0, 0] = source_crop_size / image_w
    theta[:, 1, 1] = source_crop_size / image_h
    theta[:, 0, 2] = -1.0 + (2.0 * left + source_crop_size) / image_w
    theta[:, 1, 2] = -1.0 + (2.0 * top + source_crop_size) / image_h

    grid = F.affine_grid(
        theta,
        torch.Size((batch_size, channels, output_size, output_size)),
        align_corners=False,
    )
    return F.grid_sample(images, grid, align_corners=False)


class AnnealedVanillaSaccadeImageEncoder(VanillaSaccadeImageEncoder):
    """Vanilla saccades with an annealed training-time source field of view.

    The trainable architecture and evaluation path are identical to
    ``VanillaSaccadeImageEncoder``. Early in training, each predicted location
    samples a wide source region and resizes it to ``patch_size``. That source
    region shrinks geometrically to ``patch_size`` over ``anneal_steps``
    forwards. Evaluation always uses vanilla's exact hard local crops.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "resnet18",
        emb_dim: int = 256,
        patch_size: int | None = None,
        start_scale: float = 0.8,
        anneal_steps: int = 100_000,
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
        if not 0.0 < start_scale <= 1.0:
            raise ValueError("start_scale must be in (0, 1]")
        if anneal_steps < 1:
            raise ValueError("anneal_steps must be positive")

        super().__init__(
            shape_meta=shape_meta,
            rgb_model_name=rgb_model_name,
            emb_dim=emb_dim,
            patch_size=patch_size,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
            use_seq=use_seq,
            keep_horizon_dims=keep_horizon_dims,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
        )
        self.start_scale = float(start_scale)
        self.anneal_steps = int(anneal_steps)
        self.register_buffer("anneal_step", torch.tensor(0, dtype=torch.long))

    def set_anneal_step(self, step: int) -> None:
        """Set curriculum progress explicitly when resuming training."""
        self.anneal_step.fill_(max(0, int(step)))

    def _anneal_fraction(self) -> float:
        if not self.training:
            return 1.0
        return min(float(self.anneal_step.item()) / self.anneal_steps, 1.0)

    def _source_crop_size(self, image_h: int, image_w: int) -> float:
        start_size = max(
            float(self.patch_size),
            min(image_h, image_w) * self.start_scale,
        )
        fraction = self._anneal_fraction()
        return start_size * math.pow(self.patch_size / start_size, fraction)

    def _crop(self, image: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super()._crop(image, location)

        source_crop_size = self._source_crop_size(*image.shape[-2:])
        return soft_crop_patches_with_fov(
            image,
            location,
            source_crop_size=source_crop_size,
            output_size=self.patch_size,
        )

    def forward(self, obs_dict, mask=None):
        encoded = super().forward(obs_dict, mask=mask)
        if self.training:
            self.anneal_step.add_(1)
        return encoded
