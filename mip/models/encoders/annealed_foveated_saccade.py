"""Gaussian-foveated saccades with an annealed circular field of view."""

from __future__ import annotations

import math

import torch

from mip.models.encoders.foveated_saccade import FoveatedSaccadeImageEncoder
from mip.models.encoders.foveation import _positive_finite, _positive_integer


class AnnealedFoveatedSaccadeImageEncoder(FoveatedSaccadeImageEncoder):
    """Anneal source radius while keeping output shape and foveation fixed.

    Progress advances once per successful training encoder forward, matching the
    existing annealed vanilla encoder. Evaluation always uses the final radius
    and retains Gaussian foveation. The progress buffer is checkpointed.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "resnet18",
        emb_dim: int = 256,
        patch_size: int | None = None,
        radius: float | None = None,
        start_radius: float | None = None,
        anneal_steps: int = 100_000,
        fovea_ratio: float = 0.4,
        blur_sigma: float = 2.0,
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
        anneal_steps = _positive_integer(anneal_steps, "anneal_steps")
        if start_radius is not None:
            start_radius = _positive_finite(start_radius, "start_radius")
        super().__init__(
            shape_meta=shape_meta,
            rgb_model_name=rgb_model_name,
            emb_dim=emb_dim,
            patch_size=patch_size,
            radius=radius,
            fovea_ratio=fovea_ratio,
            blur_sigma=blur_sigma,
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
        self.start_radius = 2.0 * self.radius if start_radius is None else start_radius
        self._validate_radius(self.start_radius, "start_radius")
        if self.start_radius < self.radius:
            raise ValueError("start_radius must be greater than or equal to radius")
        self.anneal_steps = anneal_steps
        self.register_buffer("anneal_step", torch.tensor(0, dtype=torch.long))

    def set_anneal_step(self, step: int) -> None:
        """Set saved curriculum progress explicitly when resuming training."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self.anneal_step.fill_(step)

    def _current_radius(self) -> float:
        if not self.training:
            return self.radius
        fraction = min(float(self.anneal_step.item()) / self.anneal_steps, 1.0)
        if fraction >= 1.0:
            return self.radius
        return self.start_radius * math.pow(self.radius / self.start_radius, fraction)

    def forward(self, obs_dict, mask=None):
        encoded = super().forward(obs_dict, mask=mask)
        if self.training:
            self.anneal_step.add_(1)
        return encoded
