"""Segmentation losses for EReCu-Med training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _ensure_label_shape(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0]
    if label.ndim != 3:
        raise ValueError(f"Expected label shape [B, H, W] or [B, 1, H, W], got {tuple(label.shape)}.")
    return label.long()


def _resize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _one_hot_valid(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = _ensure_label_shape(target)
    valid = (target != int(ignore_index)) & (target >= 0) & (target < int(num_classes))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, num_classes=int(num_classes)).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()
    return one_hot, valid.float()


def partial_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = _ensure_label_shape(target).to(device=logits.device)
    logits = _resize_logits(logits, target)
    valid = target != int(ignore_index)
    if not valid.any():
        return logits.sum() * 0.0
    loss_map = F.cross_entropy(logits, target, ignore_index=int(ignore_index), reduction="none")
    return loss_map[valid].sum() / valid.float().sum().clamp_min(eps)


def partial_dice_loss(
    probs_or_logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 4,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = _ensure_label_shape(target).to(device=probs_or_logits.device)
    probs_or_logits = _resize_logits(probs_or_logits, target)
    probs = torch.softmax(probs_or_logits, dim=1) if from_logits else probs_or_logits
    num_classes = probs.shape[1]
    target_oh, valid = _one_hot_valid(target, num_classes, ignore_index)
    valid = valid.unsqueeze(1)

    intersection = (probs * target_oh * valid).sum(dim=(0, 2, 3))
    pred_sum = (probs * probs * valid).sum(dim=(0, 2, 3))
    target_sum = (target_oh * target_oh * valid).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    present = target_oh.sum(dim=(0, 2, 3)) > 0
    if not present.any():
        return probs.sum() * 0.0
    return (1.0 - dice[present]).mean()


def partial_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 4,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    loss_ce = partial_cross_entropy(logits, target, ignore_index=ignore_index)
    loss_dice = partial_dice_loss(logits, target, ignore_index=ignore_index, from_logits=True)
    return float(ce_weight) * loss_ce + float(dice_weight) * loss_dice


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    ignore_index: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = _ensure_label_shape(target).to(device=logits.device)
    logits = _resize_logits(logits, target)
    if weight_map.ndim == 4 and weight_map.shape[1] == 1:
        weight_map = weight_map[:, 0]
    if weight_map.ndim != 3:
        raise ValueError(f"Expected weight_map [B, H, W] or [B, 1, H, W], got {tuple(weight_map.shape)}.")
    if weight_map.shape[-2:] != target.shape[-2:]:
        weight_map = F.interpolate(weight_map.unsqueeze(1).float(), size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    weight_map = weight_map.to(device=logits.device, dtype=logits.dtype)

    valid = (target != int(ignore_index)).float()
    loss_map = F.cross_entropy(logits, target, ignore_index=int(ignore_index), reduction="none")
    weights = weight_map * valid
    return (loss_map * weights).sum() / weights.sum().clamp_min(eps)


def weighted_dice_loss(
    probs_or_logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    ignore_index: int = 4,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = _ensure_label_shape(target).to(device=probs_or_logits.device)
    probs_or_logits = _resize_logits(probs_or_logits, target)
    probs = torch.softmax(probs_or_logits, dim=1) if from_logits else probs_or_logits

    if weight_map.ndim == 4 and weight_map.shape[1] == 1:
        weight_map = weight_map[:, 0]
    if weight_map.ndim != 3:
        raise ValueError(f"Expected weight_map [B, H, W] or [B, 1, H, W], got {tuple(weight_map.shape)}.")
    if weight_map.shape[-2:] != target.shape[-2:]:
        weight_map = F.interpolate(weight_map.unsqueeze(1).float(), size=target.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    weight = weight_map.to(device=probs.device, dtype=probs.dtype).unsqueeze(1)

    num_classes = probs.shape[1]
    target_oh, valid = _one_hot_valid(target, num_classes, ignore_index)
    valid_weight = weight * valid.unsqueeze(1)

    intersection = (probs * target_oh * valid_weight).sum(dim=(0, 2, 3))
    pred_sum = (probs * probs * valid_weight).sum(dim=(0, 2, 3))
    target_sum = (target_oh * target_oh * valid_weight).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    present = (target_oh * valid_weight).sum(dim=(0, 2, 3)) > 0
    if not present.any():
        return probs.sum() * 0.0
    return (1.0 - dice[present]).mean()


def weighted_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    ignore_index: int = 4,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    loss_ce = weighted_cross_entropy(logits, target, weight_map, ignore_index=ignore_index)
    loss_dice = weighted_dice_loss(logits, target, weight_map, ignore_index=ignore_index, from_logits=True)
    return float(ce_weight) * loss_ce + float(dice_weight) * loss_dice


def soft_dice_loss(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    input_from_logits: bool = True,
    target_from_logits: bool = True,
    weight_map: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target_tensor = _resize_logits(target_tensor, input_tensor)
    input_prob = torch.softmax(input_tensor, dim=1) if input_from_logits else input_tensor
    target_prob = torch.softmax(target_tensor, dim=1) if target_from_logits else target_tensor
    target_prob = target_prob.detach()

    if weight_map is None:
        weight = torch.ones_like(input_prob[:, :1])
    else:
        if weight_map.ndim == 3:
            weight = weight_map.unsqueeze(1)
        elif weight_map.ndim == 4 and weight_map.shape[1] == 1:
            weight = weight_map
        else:
            raise ValueError(f"Expected weight_map [B, H, W] or [B, 1, H, W], got {tuple(weight_map.shape)}.")
        if weight.shape[-2:] != input_prob.shape[-2:]:
            weight = F.interpolate(weight.float(), size=input_prob.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight.to(device=input_prob.device, dtype=input_prob.dtype)

    intersection = (input_prob * target_prob * weight).sum(dim=(0, 2, 3))
    input_sum = (input_prob * input_prob * weight).sum(dim=(0, 2, 3))
    target_sum = (target_prob * target_prob * weight).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (input_sum + target_sum + eps)
    return (1.0 - dice).mean()


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
    eps: float = 1e-6,
) -> torch.Tensor:
    teacher_logits = _resize_logits(teacher_logits, student_logits)
    student_prob = torch.softmax(student_logits, dim=1)
    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)

    loss_type = loss_type.lower()
    if loss_type == "mse":
        loss_map = ((student_prob - teacher_prob) ** 2).mean(dim=1)
    elif loss_type == "kl":
        loss_map = F.kl_div(
            torch.log(student_prob.clamp_min(eps)),
            teacher_prob,
            reduction="none",
        ).sum(dim=1)
    else:
        raise ValueError(f"Unsupported consistency loss_type: {loss_type}.")

    if weight_map is None:
        return loss_map.mean()
    if weight_map.ndim == 4 and weight_map.shape[1] == 1:
        weight_map = weight_map[:, 0]
    if weight_map.ndim != 3:
        raise ValueError(f"Expected weight_map [B, H, W] or [B, 1, H, W], got {tuple(weight_map.shape)}.")
    if weight_map.shape[-2:] != loss_map.shape[-2:]:
        weight_map = F.interpolate(weight_map.unsqueeze(1).float(), size=loss_map.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    weight_map = weight_map.to(device=loss_map.device, dtype=loss_map.dtype)
    return (loss_map * weight_map).sum() / weight_map.sum().clamp_min(eps)


__all__ = [
    "consistency_loss",
    "partial_ce_dice_loss",
    "partial_cross_entropy",
    "partial_dice_loss",
    "soft_dice_loss",
    "weighted_ce_dice_loss",
    "weighted_cross_entropy",
    "weighted_dice_loss",
]
