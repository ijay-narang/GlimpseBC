"""Saccade encoder with a fixed circular Gaussian-foveated field of view."""

from __future__ import annotations

import torch

from mip.models.encoders.foveation import (
    _positive_finite,
    _positive_integer,
    _validate_fovea_ratio,
    gaussian_foveated_crop,
)
from mip.models.encoders.vanilla_saccade import VanillaSaccadeImageEncoder


class FoveatedSaccadeImageEncoder(VanillaSaccadeImageEncoder):
    """Keep the vanilla network and replace only its local viewing operation.

    ``patch_size`` specifies the tensor presented to the ResNet. ``radius``
    specifies source-image coverage after optional image resizing, independently
    of that tensor size. Training and evaluation both use circular foveation.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "resnet18",
        emb_dim: int = 256,
        patch_size: int | None = None,
        radius: float | None = None,
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
        if patch_size is not None:
            _positive_integer(patch_size, "patch_size")
        fovea_ratio = _validate_fovea_ratio(fovea_ratio)
        blur_sigma = _positive_finite(blur_sigma, "blur_sigma")
        if radius is not None:
            radius = _positive_finite(radius, "radius")
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
        _positive_integer(self.patch_size, "patch_size")
        self.radius = self.patch_size / 2.0 if radius is None else radius
        self.fovea_ratio = fovea_ratio
        self.blur_sigma = blur_sigma
        self._validate_radius(self.radius, "radius")

    def _validate_radius(self, radius: float, name: str) -> None:
        """Reject fields that cannot fit any configured, prepared camera image."""
        _positive_finite(radius, name)
        for key in self.rgb_keys:
            image_shape = self.resize_shape or tuple(
                self.shape_meta["obs"][key]["shape"][-2:]
            )
            if len(image_shape) != 2 or min(image_shape) < 1:
                raise ValueError(
                    f"Invalid prepared image shape for {key}: {image_shape}"
                )
            if radius * 2.0 > min(image_shape):
                raise ValueError(
                    f"{name} {radius} does not fit camera {key!r} "
                    f"with prepared shape {image_shape}"
                )

    def _current_radius(self) -> float:
        return self.radius

    def _crop(self, image: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        return gaussian_foveated_crop(
            image,
            location,
            radius=self._current_radius(),
            output_size=self.patch_size,
            fovea_ratio=self.fovea_ratio,
            blur_sigma=self.blur_sigma,
        )
