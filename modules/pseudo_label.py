"""Pseudo-label construction for EReCu-Med.

This module turns evolved/fused soft predictions into hard pseudo labels and
per-pixel weights. Scribble labels are hard constraints; unreliable unlabeled
pixels are assigned ignore_index.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _ensure_label_shape(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0]
    if label.ndim != 3:
        raise ValueError(f"Expected label shape [B, H, W] or [B, 1, H, W], got {tuple(label.shape)}.")
    return label.long()


def _resize_map(x: torch.Tensor, target_size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == target_size:
        return x
    if mode in ("nearest", "area"):
        return F.interpolate(x, size=target_size, mode=mode)
    return F.interpolate(x, size=target_size, mode=mode, align_corners=False)


def _extract_cmnp_reliability(
    cmnp_info: Optional[Dict[str, torch.Tensor]],
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_classes, height, width = probs.shape
    device = probs.device
    dtype = probs.dtype

    if cmnp_info is None:
        pixel_reliability = torch.ones(batch_size, height, width, device=device, dtype=dtype)
        class_quality = torch.ones(batch_size, num_classes, device=device, dtype=dtype)
        return pixel_reliability, class_quality

    pixel_reliability = cmnp_info.get("pixel_reliability")
    if pixel_reliability is None:
        pixel_reliability = torch.ones(batch_size, height, width, device=device, dtype=dtype)
    else:
        if pixel_reliability.ndim == 4 and pixel_reliability.shape[1] == 1:
            pixel_reliability = pixel_reliability[:, 0]
        if pixel_reliability.ndim != 3:
            raise ValueError(
                "cmnp_info['pixel_reliability'] must have shape [B, H, W] or [B, 1, H, W]."
            )
        if pixel_reliability.shape[-2:] != (height, width):
            pixel_reliability = _resize_map(pixel_reliability.unsqueeze(1), (height, width))[:, 0]
        pixel_reliability = pixel_reliability.to(device=device, dtype=dtype)

    class_quality = cmnp_info.get("class_quality")
    if class_quality is None:
        class_quality = torch.ones(batch_size, num_classes, device=device, dtype=dtype)
    else:
        if class_quality.shape != (batch_size, num_classes):
            raise ValueError(
                f"cmnp_info['class_quality'] must have shape {(batch_size, num_classes)}, "
                f"got {tuple(class_quality.shape)}."
            )
        class_quality = class_quality.to(device=device, dtype=dtype)

    return pixel_reliability.clamp(0.0, 1.0), class_quality.clamp(0.0, 1.0)


def build_evolved_pseudo_label(
    probs_evo: torch.Tensor,
    scribble: torch.Tensor,
    cmnp_info: Optional[Dict[str, torch.Tensor]] = None,
    ignore_index: int = 4,
    from_logits: bool = False,
    confidence_threshold: float = 0.65,
    reliability_threshold: float = 0.35,
    class_quality_threshold: float = 0.0,
    min_weight: float = 0.0,
    scribble_weight: float = 1.0,
    detach: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build hard pseudo labels and pixel weights.

    Args:
        probs_evo: Evolved probabilities or logits [B, C, H, W].
        scribble: Scribble label [B, H, W] where ignore_index denotes UA.
        cmnp_info: Optional CMNP output dictionary.
        ignore_index: Unreliable/UA target value.
        from_logits: Apply softmax to probs_evo.
        confidence_threshold: Minimum max probability for unlabeled pixels.
        reliability_threshold: Minimum CMNP pixel reliability for unlabeled pixels.
        class_quality_threshold: Minimum CMNP class quality for predicted class.
        min_weight: Clamp pseudo weights below this value to zero.
        scribble_weight: Weight assigned to scribble pixels.
        detach: Detach returned targets/weights from graph.
    """
    if probs_evo.ndim != 4:
        raise ValueError(f"Expected probs/logits [B, C, H, W], got {tuple(probs_evo.shape)}.")
    probs = torch.softmax(probs_evo, dim=1) if from_logits else probs_evo
    probs = probs.clamp_min(1e-6)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

    batch_size, num_classes, height, width = probs.shape
    scribble = _ensure_label_shape(scribble).to(device=probs.device)
    if scribble.shape[-2:] != (height, width):
        scribble = _resize_map(scribble.unsqueeze(1).float(), (height, width), mode="nearest")[:, 0].long()

    conf, pred = probs.max(dim=1)
    pixel_reliability, class_quality = _extract_cmnp_reliability(cmnp_info, probs)
    pred_class_quality = class_quality.gather(1, pred.flatten(1)).view_as(conf)

    valid_scribble = (scribble != int(ignore_index)) & (scribble >= 0) & (scribble < num_classes)
    reliable = (
        (conf >= float(confidence_threshold))
        & (pixel_reliability >= float(reliability_threshold))
        & (pred_class_quality >= float(class_quality_threshold))
    )

    pseudo_label = torch.full_like(pred, int(ignore_index), dtype=torch.long)
    pseudo_label = torch.where(reliable, pred.long(), pseudo_label)
    pseudo_label = torch.where(valid_scribble, scribble.long(), pseudo_label)

    pseudo_weight = conf * pixel_reliability * pred_class_quality
    pseudo_weight = torch.where(reliable, pseudo_weight, torch.zeros_like(pseudo_weight))
    pseudo_weight = torch.where(valid_scribble, torch.full_like(pseudo_weight, float(scribble_weight)), pseudo_weight)
    if min_weight > 0.0:
        pseudo_weight = torch.where(
            pseudo_weight >= float(min_weight),
            pseudo_weight,
            torch.zeros_like(pseudo_weight),
        )
    pseudo_weight = pseudo_weight.clamp(0.0, 1.0)

    if detach:
        pseudo_label = pseudo_label.detach()
        pseudo_weight = pseudo_weight.detach()
    return pseudo_label, pseudo_weight


def pseudo_label_stats(
    pseudo_label: torch.Tensor,
    pseudo_weight: Optional[torch.Tensor] = None,
    ignore_index: int = 4,
    num_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return lightweight diagnostics for logging."""
    pseudo_label = _ensure_label_shape(pseudo_label)
    valid = pseudo_label != int(ignore_index)
    total = torch.tensor(float(pseudo_label.numel()), device=pseudo_label.device)
    valid_ratio = valid.float().sum() / total.clamp_min(1.0)

    stats: Dict[str, torch.Tensor] = {"valid_ratio": valid_ratio.detach()}
    if pseudo_weight is not None:
        if pseudo_weight.ndim == 4 and pseudo_weight.shape[1] == 1:
            pseudo_weight = pseudo_weight[:, 0]
        stats["mean_weight"] = pseudo_weight.float().mean().detach()
        stats["valid_mean_weight"] = pseudo_weight.float()[valid].mean().detach() if valid.any() else pseudo_weight.new_zeros(())

    if num_classes is not None:
        counts = []
        for class_idx in range(int(num_classes)):
            counts.append((pseudo_label == class_idx).float().sum())
        stats["class_counts"] = torch.stack(counts).detach()
    return stats


__all__ = ["build_evolved_pseudo_label", "pseudo_label_stats"]
