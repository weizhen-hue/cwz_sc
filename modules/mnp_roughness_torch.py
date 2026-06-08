"""DoG + LBP + roughness native-cue extractor for CMNP.

SPSD summarizes patch-level spectral/fractal structure. On Prostate MRI this
can be too coarse for the PZ/CG boundary. This extractor replaces SPSD with
pixel-aligned multi-scale roughness cues:

    local intensity standard deviation
    local high-pass residual energy
    Laplacian magnitude

The output keeps the same role as the other MNP extractors: it provides fixed
native image features for CMNP and does not consume predictions.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mnp_spsd_torch import (
    compute_dog,
    compute_lbp,
    gaussian_blur2d,
    gradient_magnitude,
    normalize_minmax,
    standardize_channels,
)


def laplacian_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Depthwise 3x3 Laplacian magnitude for [B, 1, H, W]."""
    if image.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], got {tuple(image.shape)}.")
    channels = image.shape[1]
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels).abs()


class MNPRoughnessFeatureExtractor(nn.Module):
    """Build DoG + LBP + local roughness features for CMNP.

    Args:
        dog_sigmas: Gaussian scales for Difference-of-Gaussians.
        roughness_sigmas: Gaussian scales for local roughness estimation.
        detach_output: If True, native cue tensors do not carry gradients.

    Input:
        image: [B, 1, H, W] or multi-channel image.

    Output:
        f_mnp: [B, 2 + 2 * len(roughness_sigmas) + 1, H, W].
            Channels are [LBP, DoG_abs, local_std..., residual_energy..., laplacian].
        edge_map: [B, 1, H, W], same DoG/gradient edge cue used by the other variants.
    """

    def __init__(
        self,
        dog_sigmas: Tuple[float, float] = (1.0, 2.5),
        roughness_sigmas: Sequence[float] = (1.0, 2.0, 4.0),
        roughness_weight: float = 0.35,
        detach_output: bool = True,
    ) -> None:
        super().__init__()
        if len(roughness_sigmas) < 1:
            raise ValueError("roughness_sigmas must contain at least one scale.")
        self.dog_sigmas = (float(dog_sigmas[0]), float(dog_sigmas[1]))
        self.roughness_sigmas = tuple(float(v) for v in roughness_sigmas)
        self.roughness_weight = float(roughness_weight)
        self.detach_output = bool(detach_output)

    @property
    def out_channels(self) -> int:
        return 2 + 2 * len(self.roughness_sigmas) + 1

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(image.shape)}.")
        if image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)

        with torch.set_grad_enabled(not self.detach_output and torch.is_grad_enabled()):
            image = normalize_minmax(image.float())
            _, dog_abs = compute_dog(
                image,
                sigma_small=self.dog_sigmas[0],
                sigma_large=self.dog_sigmas[1],
            )
            dog_ch = normalize_minmax(dog_abs)
            lbp_ch = compute_lbp(image)
            roughness = self._roughness_features(image)
            roughness = standardize_channels(roughness) * self.roughness_weight
            f_mnp = torch.cat([lbp_ch, dog_ch, roughness], dim=1)
            edge_map = normalize_minmax(0.5 * dog_abs + 0.5 * gradient_magnitude(image))

        if self.detach_output:
            return f_mnp.detach(), edge_map.detach()
        return f_mnp, edge_map

    def _roughness_features(self, image: torch.Tensor) -> torch.Tensor:
        image01 = normalize_minmax(image)
        local_std_maps = []
        residual_maps = []

        for sigma in self.roughness_sigmas:
            mean = gaussian_blur2d(image01, sigma=sigma)
            mean_sq = gaussian_blur2d(image01 * image01, sigma=sigma)
            local_var = (mean_sq - mean * mean).clamp_min(0.0)
            local_std_maps.append(normalize_minmax(torch.sqrt(local_var + 1e-12)))

            residual = (image01 - mean).abs()
            residual = gaussian_blur2d(residual, sigma=max(0.5, 0.5 * sigma))
            residual_maps.append(normalize_minmax(residual))

        lap = normalize_minmax(laplacian_magnitude(image01))
        return torch.cat([*local_std_maps, *residual_maps, lap], dim=1)


__all__ = ["MNPRoughnessFeatureExtractor", "laplacian_magnitude"]
