"""DoG + LBP native-cue extractor for the no-SPSD CMNP ablation.

This module is intentionally narrower than ``mnp_spsd_torch``. It keeps the
same DoG/gradient edge map used by the full version, but removes the SPSD
embedding from the feature tensor. That isolates the effect of SPSD on class
prototype quality and pseudo-label reliability.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .mnp_spsd_torch import compute_dog, compute_lbp, gradient_magnitude, normalize_minmax


class MNPDogLBPFeatureExtractor(nn.Module):
    """Build DoG + LBP native features for CMNP without SPSD channels.

    Args:
        dog_sigmas: Gaussian scales for Difference-of-Gaussians.
        detach_output: If True, native cue tensors do not carry gradients.

    Input:
        image: [B, 1, H, W] or multi-channel image. Multi-channel input is
            averaged to one channel for cue extraction.

    Output:
        f_mnp: [B, 2, H, W], channel order [LBP, DoG_abs].
        edge_map: [B, 1, H, W], same edge cue as the full SPSD version.
    """

    def __init__(
        self,
        dog_sigmas: Tuple[float, float] = (1.0, 2.5),
        detach_output: bool = True,
    ) -> None:
        super().__init__()
        self.dog_sigmas = (float(dog_sigmas[0]), float(dog_sigmas[1]))
        self.detach_output = bool(detach_output)

    @property
    def out_channels(self) -> int:
        return 2

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(image.shape)}.")
        if image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)

        with torch.set_grad_enabled(not self.detach_output and torch.is_grad_enabled()):
            image = image.float()
            _, dog_abs = compute_dog(
                image,
                sigma_small=self.dog_sigmas[0],
                sigma_large=self.dog_sigmas[1],
            )
            dog_ch = normalize_minmax(dog_abs)
            lbp_ch = compute_lbp(image)
            f_mnp = torch.cat([lbp_ch, dog_ch], dim=1)
            edge_map = normalize_minmax(0.5 * dog_abs + 0.5 * gradient_magnitude(image))

        if self.detach_output:
            return f_mnp.detach(), edge_map.detach()
        return f_mnp, edge_map


__all__ = ["MNPDogLBPFeatureExtractor"]
