"""Torch native-cue extractor for EReCu-Med MNP.

This module replaces the original EReCu ResNet18-MNP semantic branch with
training-time native image cues: DoG, LBP, and SPSD-like patch descriptors.
It does not consume predictions and does not compute CMNP losses.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


def normalize_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each sample/channel to [0, 1]."""
    reduce_dims = tuple(range(2, x.ndim))
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def standardize_channels(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize each sample/channel over spatial dimensions."""
    reduce_dims = tuple(range(2, x.ndim))
    mean = x.mean(dim=reduce_dims, keepdim=True)
    std = x.std(dim=reduce_dims, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def gaussian_kernel1d(
    sigma: float,
    truncate: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return torch.ones(1, device=device, dtype=dtype or torch.float32)
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-6)


def gaussian_blur2d(x: torch.Tensor, sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Depthwise separable Gaussian blur for [B, C, H, W]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got shape {tuple(x.shape)}.")
    kernel = gaussian_kernel1d(sigma, truncate=truncate, device=x.device, dtype=x.dtype)
    pad = kernel.numel() // 2
    if pad == 0:
        return x

    channels = x.shape[1]
    kernel_h = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    kernel_w = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)

    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel_h, groups=channels)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_w, groups=channels)
    return x


def compute_dog(
    image: torch.Tensor,
    sigma_small: float = 1.0,
    sigma_large: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image01 = normalize_minmax(image)
    small = gaussian_blur2d(image01, sigma_small)
    large = gaussian_blur2d(image01, sigma_large)
    signed = small - large
    return signed, signed.abs()


def compute_lbp(image: torch.Tensor) -> torch.Tensor:
    """Compute normalized 8-neighbor LBP code for [B, 1, H, W]."""
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"LBP expects [B, 1, H, W], got {tuple(image.shape)}.")
    image01 = normalize_minmax(image)
    padded = F.pad(image01, (1, 1, 1, 1), mode="replicate")
    center = padded[:, :, 1:-1, 1:-1]
    neighbors = (
        padded[:, :, :-2, :-2],
        padded[:, :, :-2, 1:-1],
        padded[:, :, :-2, 2:],
        padded[:, :, 1:-1, 2:],
        padded[:, :, 2:, 2:],
        padded[:, :, 2:, 1:-1],
        padded[:, :, 2:, :-2],
        padded[:, :, 1:-1, :-2],
    )

    code = torch.zeros_like(center)
    for bit, neighbor in enumerate(neighbors):
        code = code + (neighbor >= center).to(image.dtype) * float(1 << bit)
    return code / 255.0


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    image01 = normalize_minmax(image)
    gx = torch.zeros_like(image01)
    gy = torch.zeros_like(image01)
    gx[:, :, :, 1:-1] = 0.5 * (image01[:, :, :, 2:] - image01[:, :, :, :-2])
    gy[:, :, 1:-1, :] = 0.5 * (image01[:, :, 2:, :] - image01[:, :, :-2, :])
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


class MNPSPSDFeatureExtractor(nn.Module):
    """Build DoG + LBP + SPSD native features for CMNP.

    Args:
        spsd_scales: Gaussian scales used for Holder-like slope estimation.
        spsd_patch_size: Patch size for SPSD histograms.
        spsd_stride: Patch stride for SPSD histograms.
        spsd_bins: Number of soft Holder bins.
        spsd_embed_dim: Fixed projected SPSD channel count.
    """

    def __init__(
        self,
        dog_sigmas: Tuple[float, float] = (1.0, 2.5),
        spsd_scales: Sequence[float] = (1.0, 2.0, 4.0),
        spsd_patch_size: int = 16,
        spsd_stride: int = 16,
        spsd_bins: int = 12,
        spsd_embed_dim: int = 16,
        projection_seed: int = 2026,
        detach_output: bool = True,
    ) -> None:
        super().__init__()
        if len(spsd_scales) < 2:
            raise ValueError("SPSD requires at least two scales.")
        if spsd_patch_size <= 0 or spsd_stride <= 0:
            raise ValueError("SPSD patch size and stride must be positive.")

        self.dog_sigmas = (float(dog_sigmas[0]), float(dog_sigmas[1]))
        self.spsd_scales = _as_tuple(spsd_scales)
        self.spsd_patch_size = int(spsd_patch_size)
        self.spsd_stride = int(spsd_stride)
        self.spsd_bins = int(spsd_bins)
        self.spsd_embed_dim = int(spsd_embed_dim)
        self.detach_output = bool(detach_output)

        generator = torch.Generator()
        generator.manual_seed(int(projection_seed))
        proj = torch.randn(self.spsd_bins, self.spsd_embed_dim, generator=generator)
        proj = proj / math.sqrt(float(max(1, self.spsd_bins)))
        self.register_buffer("spsd_projection", proj.float(), persistent=False)

        centers = torch.linspace(0.0, 1.0, self.spsd_bins)
        self.register_buffer("spsd_bin_centers", centers.view(1, self.spsd_bins, 1, 1), persistent=False)

    @property
    def out_channels(self) -> int:
        return 2 + self.spsd_embed_dim

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(image.shape)}.")
        if image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)

        with torch.set_grad_enabled(not self.detach_output and torch.is_grad_enabled()):
            image = image.float()
            _, dog_abs = compute_dog(image, sigma_small=self.dog_sigmas[0], sigma_large=self.dog_sigmas[1])
            dog_ch = normalize_minmax(dog_abs)
            lbp_ch = compute_lbp(image)
            holder_map = self._estimate_holder_map(image)
            energy_map = normalize_minmax(dog_abs + gradient_magnitude(image))
            z_spsd = self._spsd_embedding(holder_map, energy_map, image.shape[-2:])
            z_spsd = standardize_channels(z_spsd)

            f_mnp = torch.cat([lbp_ch, dog_ch, z_spsd], dim=1)
            edge_map = normalize_minmax(0.5 * dog_abs + 0.5 * gradient_magnitude(image))

        if self.detach_output:
            return f_mnp.detach(), edge_map.detach()
        return f_mnp, edge_map

    def _estimate_holder_map(self, image: torch.Tensor) -> torch.Tensor:
        image01 = normalize_minmax(image)
        responses = []
        for sigma in self.spsd_scales:
            detail = (image01 - gaussian_blur2d(image01, sigma)).abs()
            smooth_detail = gaussian_blur2d(detail, max(0.5, sigma * 0.5))
            responses.append(torch.log(smooth_detail + 1e-6))

        y = torch.stack(responses, dim=0)
        x = torch.tensor(self.spsd_scales, device=image.device, dtype=image.dtype).log()
        x_centered = x - x.mean()
        denom = (x_centered * x_centered).sum().clamp_min(1e-6)
        y_centered = y - y.mean(dim=0, keepdim=True)
        slope = (y_centered * x_centered.view(-1, 1, 1, 1, 1)).sum(dim=0) / denom
        return normalize_minmax(slope)

    def _spsd_embedding(
        self,
        holder_map: torch.Tensor,
        energy_map: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        bandwidth = 1.0 / float(max(2, self.spsd_bins - 1))
        holder = holder_map.clamp(0.0, 1.0)
        centers = self.spsd_bin_centers.to(device=holder.device, dtype=holder.dtype)
        soft_bins = torch.exp(-0.5 * ((holder - centers) / bandwidth) ** 2)
        soft_bins = soft_bins / soft_bins.sum(dim=1, keepdim=True).clamp_min(1e-6)

        weighted_energy = soft_bins * energy_map
        hist = F.avg_pool2d(
            weighted_energy,
            kernel_size=self.spsd_patch_size,
            stride=self.spsd_stride,
            ceil_mode=True,
        )
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)

        proj = self.spsd_projection.to(device=hist.device, dtype=hist.dtype)
        z = torch.einsum("bkhw,kd->bdhw", hist, proj)
        return F.interpolate(z, size=image_size, mode="bilinear", align_corners=False)


__all__ = [
    "MNPSPSDFeatureExtractor",
    "compute_dog",
    "compute_lbp",
    "gaussian_blur2d",
    "gradient_magnitude",
    "normalize_minmax",
    "standardize_channels",
]
