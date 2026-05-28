"""Pseudo-label Evolution Fusion for EReCu-Med.

This module fuses Student, EMA Teacher, DSC, and STAF predictions into an
evolved soft prediction. It keeps the operation differentiability optional and
does not build hard pseudo labels; use pseudo_label.py for that step.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_probs(x: torch.Tensor, from_logits: bool) -> torch.Tensor:
    probs = torch.softmax(x, dim=1) if from_logits else x
    probs = probs.clamp_min(1e-6)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _entropy_confidence(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = probs.shape[1]
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
    entropy = entropy / torch.log(torch.tensor(float(num_classes), device=probs.device, dtype=probs.dtype))
    return (1.0 - entropy).clamp(0.0, 1.0)


def _prediction_confidence(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)


def _cmnp_pixel_reliability(
    cmnp_info: Optional[Dict[str, torch.Tensor]],
    ref: torch.Tensor,
) -> torch.Tensor:
    if cmnp_info is None or cmnp_info.get("pixel_reliability") is None:
        return torch.ones(ref.shape[0], 1, ref.shape[2], ref.shape[3], device=ref.device, dtype=ref.dtype)
    reliability = cmnp_info["pixel_reliability"].to(device=ref.device, dtype=ref.dtype)
    if reliability.ndim == 3:
        reliability = reliability.unsqueeze(1)
    if reliability.ndim != 4 or reliability.shape[1] != 1:
        raise ValueError("cmnp_info['pixel_reliability'] must have shape [B, H, W] or [B, 1, H, W].")
    return _resize_like(reliability, ref).clamp(0.0, 1.0)


def _cmnp_class_quality(
    cmnp_info: Optional[Dict[str, torch.Tensor]],
    ref: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_classes = ref.shape[:2]
    if cmnp_info is None or cmnp_info.get("class_quality") is None:
        return torch.ones(batch_size, num_classes, device=ref.device, dtype=ref.dtype)
    class_quality = cmnp_info["class_quality"].to(device=ref.device, dtype=ref.dtype)
    if class_quality.shape != (batch_size, num_classes):
        raise ValueError(
            f"cmnp_info['class_quality'] must have shape {(batch_size, num_classes)}, "
            f"got {tuple(class_quality.shape)}."
        )
    return class_quality.clamp(0.0, 1.0)


class PEFFusion(nn.Module):
    """Fuse EReCu-Med soft predictions into P_evo.

    Args:
        w_student: Base weight for main student prediction.
        w_teacher: Base weight for EMA teacher prediction.
        w_dsc: Base weight for EPL/DSC prediction.
        w_staf: Base weight for STAF prediction.
        use_confidence_weight: Reweight each branch by prediction confidence.
        use_cmnp_reliability: Use CMNP pixel reliability to emphasize DSC/STAF.
        detach_teacher: Stop gradients through teacher predictions.
        detach_fusion_target: Detach all branch probabilities before fusion.
    """

    def __init__(
        self,
        w_student: float = 0.25,
        w_teacher: float = 0.35,
        w_dsc: float = 0.20,
        w_staf: float = 0.20,
        use_confidence_weight: bool = True,
        use_cmnp_reliability: bool = True,
        detach_teacher: bool = True,
        detach_fusion_target: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_student = float(w_student)
        self.w_teacher = float(w_teacher)
        self.w_dsc = float(w_dsc)
        self.w_staf = float(w_staf)
        self.use_confidence_weight = bool(use_confidence_weight)
        self.use_cmnp_reliability = bool(use_cmnp_reliability)
        self.detach_teacher = bool(detach_teacher)
        self.detach_fusion_target = bool(detach_fusion_target)
        self.eps = float(eps)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        dsc_logits: Optional[torch.Tensor] = None,
        staf_logits: Optional[torch.Tensor] = None,
        cmnp_info: Optional[Dict[str, torch.Tensor]] = None,
        inputs_are_logits: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p_student = _as_probs(student_logits, from_logits=inputs_are_logits)
        p_teacher = self._prepare_optional(teacher_logits, p_student, inputs_are_logits, "teacher")
        p_dsc = self._prepare_optional(dsc_logits, p_student, inputs_are_logits, "dsc")
        p_staf = self._prepare_optional(staf_logits, p_student, inputs_are_logits, "staf")

        if self.detach_teacher and p_teacher is not None:
            p_teacher = p_teacher.detach()

        if self.detach_fusion_target:
            p_student = p_student.detach()
            p_dsc = p_dsc.detach() if p_dsc is not None else None
            p_staf = p_staf.detach() if p_staf is not None else None
            p_teacher = p_teacher.detach() if p_teacher is not None else None

        reliability = _cmnp_pixel_reliability(cmnp_info, p_student)
        class_quality = _cmnp_class_quality(cmnp_info, p_student)
        class_quality_map = class_quality[:, :, None, None]

        branch_probs = []
        branch_weights = []

        self._append_branch(branch_probs, branch_weights, p_student, self.w_student, None)
        self._append_branch(branch_probs, branch_weights, p_teacher, self.w_teacher, None)
        self._append_branch(
            branch_probs,
            branch_weights,
            p_dsc,
            self.w_dsc,
            reliability if self.use_cmnp_reliability else None,
        )
        self._append_branch(
            branch_probs,
            branch_weights,
            p_staf,
            self.w_staf,
            reliability if self.use_cmnp_reliability else None,
        )

        if not branch_probs:
            raise ValueError("At least one prediction branch is required for PEF fusion.")

        probs_stack = torch.stack(branch_probs, dim=1)
        weights_stack = torch.stack(branch_weights, dim=1)
        weights_stack = weights_stack / weights_stack.sum(dim=1, keepdim=True).clamp_min(self.eps)
        p_evo = (probs_stack * weights_stack).sum(dim=1)
        p_evo = p_evo * class_quality_map.clamp_min(self.eps)
        p_evo = p_evo / p_evo.sum(dim=1, keepdim=True).clamp_min(self.eps)

        info = {
            "branch_weights": weights_stack.detach(),
            "pixel_reliability": reliability.detach(),
            "class_quality": class_quality.detach(),
            "teacher_available": torch.tensor(p_teacher is not None, device=p_student.device),
            "dsc_available": torch.tensor(p_dsc is not None, device=p_student.device),
            "staf_available": torch.tensor(p_staf is not None, device=p_student.device),
        }
        return p_evo, info

    def _prepare_optional(
        self,
        value: Optional[torch.Tensor],
        ref: torch.Tensor,
        inputs_are_logits: bool,
        name: str,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        probs = _as_probs(value, from_logits=inputs_are_logits)
        probs = _resize_like(probs, ref)
        if probs.shape[:2] != ref.shape[:2]:
            raise ValueError(
                f"{name} prediction must have shape [B, C, H, W] matching student; "
                f"got {tuple(probs.shape)} vs {tuple(ref.shape)}."
            )
        return probs

    def _append_branch(
        self,
        branch_probs,
        branch_weights,
        probs: Optional[torch.Tensor],
        base_weight: float,
        reliability: Optional[torch.Tensor],
    ) -> None:
        if probs is None or base_weight <= 0.0:
            return
        weight = torch.full(
            (probs.shape[0], 1, probs.shape[2], probs.shape[3]),
            float(base_weight),
            device=probs.device,
            dtype=probs.dtype,
        )
        if self.use_confidence_weight:
            weight = weight * (0.5 * _prediction_confidence(probs) + 0.5 * _entropy_confidence(probs))
        if reliability is not None:
            weight = weight * reliability
        branch_probs.append(probs)
        branch_weights.append(weight)


def fuse_predictions(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    dsc_logits: Optional[torch.Tensor] = None,
    staf_logits: Optional[torch.Tensor] = None,
    cmnp_info: Optional[Dict[str, torch.Tensor]] = None,
    inputs_are_logits: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return PEFFusion(**kwargs)(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        dsc_logits=dsc_logits,
        staf_logits=staf_logits,
        cmnp_info=cmnp_info,
        inputs_are_logits=inputs_are_logits,
    )


__all__ = ["PEFFusion", "fuse_predictions"]
