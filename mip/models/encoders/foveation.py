"""Differentiable local circular glimpses with Gaussian foveation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_fovea_ratio(value: float) -> float:
    value = _positive_finite(value, "fovea_ratio")
    if value > 1.0:
        raise ValueError("fovea_ratio must be in (0, 1]")
    return value


def _normalized_axis(size: int, reference: torch.Tensor) -> torch.Tensor:
    """Return pixel-center coordinates spanning a square of side two."""
    return (
        torch.arange(size, device=reference.device, dtype=reference.dtype) + 0.5
    ) * (2.0 / size) - 1.0


def _masked_gaussian_blur(
    image: torch.Tensor, support: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Blur only supported pixels, renormalizing at the circular boundary."""
    kernel_radius = math.ceil(3.0 * sigma)
    offsets = torch.arange(
        -kernel_radius,
        kernel_radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    # Include the support as another channel so it undergoes exactly the same
    # two filters. Dividing by its filtered value avoids a dark boundary halo.
    signals = torch.cat(
        (image * support, support.expand(image.shape[0], -1, -1, -1)), dim=1
    )
    channels = signals.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    signals = F.conv2d(signals, horizontal, padding=(0, kernel_radius), groups=channels)
    signals = F.conv2d(signals, vertical, padding=(kernel_radius, 0), groups=channels)
    denominator = signals[:, -1:].clamp_min(torch.finfo(image.dtype).eps)
    return signals[:, :-1] / denominator


def gaussian_foveated_crop(
    images: torch.Tensor,
    locations: torch.Tensor,
    radius: float,
    output_size: int,
    fovea_ratio: float = 0.4,
    blur_sigma: float = 2.0,
) -> torch.Tensor:
    """Sample a finite circular field with a sharp center and blurred periphery.

    ``images`` has shape ``[B, C, H, W]``. ``locations`` is ``[B, 2]`` in
    normalized top-left ``[y, x]`` coordinates, as in the vanilla encoder.
    A location maps the square bounding a circle of source radius ``radius``
    into the image. Source pixel centers outside that circle never contribute.

    The output is a fixed ``output_size`` square with zeroed circular corners.
    Its unwarped local image is blended with a normalized Gaussian blur using
    ``exp(-r**2 / (2 * (fovea_ratio * radius)**2))`` as the sharp-image weight.
    ``blur_sigma`` is measured in output pixels, not source pixels.

    For shrinking fields, a fixed supersampling grid covers at least one sample
    per source pixel, followed by antialiased downsampling. Antialias strength
    increases continuously from zero at native size to one at a twofold size
    reduction. Native-size and upsampled fields use direct bilinear sampling,
    preserving the sharp center without an extra filter. Both paths are
    computed at every radius, keeping operator shapes fixed throughout
    annealing. RGB and support are resampled together and normalized before
    blurring, preventing outside-image leakage and dark halos. Gradients reach
    gaze through the bilinear sampling grid; the finite source support itself
    is a hard, piecewise-constant mask.
    """
    radius = _positive_finite(radius, "radius")
    output_size = _positive_integer(output_size, "output_size")
    fovea_ratio = _validate_fovea_ratio(fovea_ratio)
    blur_sigma = _positive_finite(blur_sigma, "blur_sigma")
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError(
            "images must be a floating-point tensor with shape [B, C, H, W]"
        )
    batch_size, channels, image_h, image_w = images.shape
    if min(batch_size, channels, image_h, image_w) < 1:
        raise ValueError("images dimensions must be nonempty")
    if locations.shape != (batch_size, 2) or not locations.is_floating_point():
        raise ValueError("locations must be a floating-point tensor with shape [B, 2]")
    if locations.device != images.device:
        raise ValueError("images and locations must be on the same device")
    diameter = 2.0 * radius
    if diameter > min(image_h, image_w):
        raise ValueError(
            f"radius {radius} does not fit image shape {(image_h, image_w)}"
        )

    # Antialiased interpolation is not supported for every half-precision
    # backend. Keep filtering in float32 (or float64 for gradcheck) under AMP.
    original_dtype = images.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=images.device.type, enabled=False):
        images = images.to(work_dtype)
        locations = locations.to(work_dtype)
        center_y = locations[:, 0] * (image_h - diameter) + radius
        center_x = locations[:, 1] * (image_w - diameter) + radius
        source_y = torch.arange(image_h, device=images.device, dtype=work_dtype) + 0.5
        source_x = torch.arange(image_w, device=images.device, dtype=work_dtype) + 0.5
        distance_sq = (source_y[None, :, None] - center_y[:, None, None]).square() + (
            source_x[None, None, :] - center_x[:, None, None]
        ).square()
        source_support = (distance_sq <= radius * radius).unsqueeze(1).to(work_dtype)

        sample_size = max(output_size, min(image_h, image_w))
        sample_axis = _normalized_axis(sample_size, images)
        sample_y = center_y[:, None, None] + radius * sample_axis[None, :, None]
        sample_x = center_x[:, None, None] + radius * sample_axis[None, None, :]
        grid = torch.stack(
            (
                (2.0 * sample_x / image_w - 1.0).expand(-1, sample_size, -1),
                (2.0 * sample_y / image_h - 1.0).expand(-1, -1, sample_size),
            ),
            dim=-1,
        )
        signals = torch.cat((images * source_support, source_support), dim=1)
        sampled = F.grid_sample(signals, grid, align_corners=False)
        sample_support = (
            sample_axis[:, None].square() + sample_axis[None, :].square() <= 1.0
        ).to(work_dtype)[None, None]
        sampled = sampled * sample_support
        if sample_size != output_size:
            sampled = F.interpolate(
                sampled,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        output_axis = _normalized_axis(output_size, images)
        direct_y = center_y[:, None, None] + radius * output_axis[None, :, None]
        direct_x = center_x[:, None, None] + radius * output_axis[None, None, :]
        direct_grid = torch.stack(
            (
                (2.0 * direct_x / image_w - 1.0).expand(-1, output_size, -1),
                (2.0 * direct_y / image_h - 1.0).expand(-1, -1, output_size),
            ),
            dim=-1,
        )
        direct = F.grid_sample(signals, direct_grid, align_corners=False)
        # Blend continuously near native size: switching straight to the
        # filtered path would abruptly blur the center at radius=P/2+epsilon.
        # Arithmetic blending retains both paths in the backward graph and
        # prevents a radius-dependent FLOP count hidden by profiling caches.
        antialias_weight = min(1.0, max(0.0, diameter / output_size - 1.0))
        sampled = (1.0 - antialias_weight) * direct + antialias_weight * sampled
        sharp = sampled[:, :channels] / sampled[:, channels:].clamp_min(
            torch.finfo(work_dtype).eps
        )

        radial_sq = output_axis[:, None].square() + output_axis[None, :].square()
        output_support = (radial_sq <= 1.0).to(work_dtype)[None, None]
        sharp = sharp * output_support
        blurred = _masked_gaussian_blur(sharp, output_support, blur_sigma)
        sharp_weight = torch.exp(-radial_sq / (2.0 * fovea_ratio**2))[None, None]
        result = (
            sharp_weight * sharp + (1.0 - sharp_weight) * blurred
        ) * output_support
    return result.to(original_dtype)
