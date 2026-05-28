"""Lightweight mask heads for EReCu-style DINO features.

The original EReCu framework converts DINO features/attention into pseudo
masks through pooling, DSC, and STAF-style fusion. In the scribble medical
setting these heads output multi-class logits. Hard pseudo labels, CMNP
filtering, and scribble constraints are handled outside this module.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SizeLike = Optional[Union[torch.Tensor, Sequence[int], Tuple[int, int]]]


def _to_2tuple(size: SizeLike) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if torch.is_tensor(size):
        if size.ndim < 2:
            raise ValueError("Tensor output_size must have at least two spatial dimensions.")
        return int(size.shape[-2]), int(size.shape[-1])
    if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
        pair = tuple(size)
        if len(pair) < 2:
            raise ValueError(f"Expected at least two values for output_size, got {size}.")
        return int(pair[-2]), int(pair[-1])
    raise TypeError(f"Unsupported output_size type: {type(size)!r}.")


def _resize_logits(logits: torch.Tensor, output_size: SizeLike) -> torch.Tensor:
    target_size = _to_2tuple(output_size)
    if target_size is None or tuple(logits.shape[-2:]) == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        groups = min(32, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}.")


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "gn",
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            _make_norm(norm, out_channels),
            nn.GELU() if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CoarseMaskHead(nn.Module):
    """Minimal feature-to-mask projection for EReCu Pool/Binarize-style masks.

    This head deliberately stays shallow: it maps one DINO feature map to
    multi-class logits with a pointwise projection and optional smoothing block.
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 4,
        hidden_channels: int = 128,
        use_smoothing: bool = True,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _make_norm(norm, hidden_channels),
            nn.GELU(),
        ]
        if use_smoothing:
            layers.append(ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm))
        layers.append(nn.Conv2d(hidden_channels, num_classes, kernel_size=1))
        self.proj = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor, output_size: SizeLike = None) -> torch.Tensor:
        logits = self.proj(feature)
        return _resize_logits(logits, output_size)


class DSCHead(nn.Module):
    """Student shallow-feature head used for EPL pseudo-label evolution.

    It corresponds to the role of DSC(F_s^i) in EReCu, but outputs multi-class
    logits instead of a binarized foreground/background mask.
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 4,
        hidden_channels: int = 192,
        dropout: float = 0.0,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=1, norm=norm),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm),
            nn.Dropout2d(p=float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, output_size: SizeLike = None) -> torch.Tensor:
        logits = self.head(feature)
        return _resize_logits(logits, output_size)


class PatchSegHead(nn.Module):
    """Main segmentation projection from a DINO feature map."""

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 4,
        hidden_channels: int = 256,
        dropout: float = 0.1,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=3, norm=norm),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm),
            nn.Dropout2d(p=float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, output_size: SizeLike = None) -> torch.Tensor:
        logits = self.head(feature)
        return _resize_logits(logits, output_size)


class STAFHead(nn.Module):
    """Multi-layer student feature fusion head.

    EReCu's STAF fuses hierarchical student attention/features. This
    implementation keeps the same interface objective: selected DINO layers are
    projected to a common width, spatially aligned, weighted by a learned
    layer-attention, and converted to multi-class logits.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]] = 384,
        num_classes: int = 4,
        hidden_channels: int = 192,
        num_layers: int = 3,
        norm: str = "gn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(in_channels, int):
            channels = [int(in_channels)] * int(num_layers)
        else:
            channels = [int(ch) for ch in in_channels]
            num_layers = len(channels)
        if num_layers <= 0:
            raise ValueError("STAFHead requires at least one input layer.")

        self.num_layers = int(num_layers)
        self.projections = nn.ModuleList(
            [ConvNormAct(ch, hidden_channels, kernel_size=1, norm=norm) for ch in channels]
        )
        self.layer_score = nn.ModuleList(
            [nn.Conv2d(hidden_channels, 1, kernel_size=1) for _ in range(self.num_layers)]
        )
        self.refine = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm),
            nn.Dropout2d(p=float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor], output_size: SizeLike = None) -> torch.Tensor:
        if len(features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature maps, got {len(features)}.")

        target_grid = features[-1].shape[-2:]
        projected: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        for feature, proj, scorer in zip(features, self.projections, self.layer_score):
            feat = proj(feature)
            if feat.shape[-2:] != target_grid:
                feat = F.interpolate(feat, size=target_grid, mode="bilinear", align_corners=False)
            projected.append(feat)
            scores.append(scorer(feat))

        score_tensor = torch.stack(scores, dim=1)
        weights = torch.softmax(score_tensor, dim=1)
        feature_tensor = torch.stack(projected, dim=1)
        fused = torch.sum(weights * feature_tensor, dim=1)
        logits = self.refine(fused)
        return _resize_logits(logits, output_size)


__all__ = [
    "CoarseMaskHead",
    "ConvNormAct",
    "DSCHead",
    "PatchSegHead",
    "STAFHead",
]
