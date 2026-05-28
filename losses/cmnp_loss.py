"""Class-aware MNP loss for EReCu-Med.

The original EReCu MNP checks binary foreground/background region quality.
This module extends that idea to scribble-supervised multi-class medical
segmentation: class-wise compactness, inter-class separation, boundary
alignment, and scribble-anchored prototypes.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode in ("nearest", "area"):
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


def _ensure_label_shape(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 4 and label.shape[1] == 1:
        label = label[:, 0]
    if label.ndim != 3:
        raise ValueError(f"Expected label shape [B, H, W] or [B, 1, H, W], got {tuple(label.shape)}.")
    return label.long()


def _one_hot_ignore(
    label: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    label = _ensure_label_shape(label)
    valid = (label != int(ignore_index)) & (label >= 0) & (label < int(num_classes))
    safe_label = torch.where(valid, label, torch.zeros_like(label))
    one_hot = F.one_hot(safe_label, num_classes=int(num_classes)).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()
    return one_hot, valid


def _spatial_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, :, :, 1:] = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy[:, :, 1:, :] = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    return dx + dy


def _safe_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


class CMNPLoss(nn.Module):
    """Multi-class native perception loss.

    Args:
        num_classes: Number of semantic classes, excluding ignore_index.
        ignore_index: Scribble unlabeled area index, e.g. 4 for ACDC scribbles.
        probability_power: Sharpens class weights for prototype estimation.
        inter_margin: Minimum feature distance between class prototypes.
        min_region_weight: Minimum soft class mass required to treat a class as present.
        min_scribble_pixels: Minimum scribble pixels required for a scribble prototype.
        adjacency_matrix: Optional [C, C] weights for inter-class separation.
    """

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = 4,
        probability_power: float = 2.0,
        inter_margin: float = 0.35,
        min_region_weight: float = 8.0,
        min_scribble_pixels: float = 1.0,
        lambda_intra: float = 1.0,
        lambda_inter: float = 0.25,
        lambda_boundary: float = 0.5,
        lambda_scribble_proto: float = 0.5,
        adjacency_matrix: Optional[Sequence[Sequence[float]]] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.probability_power = float(probability_power)
        self.inter_margin = float(inter_margin)
        self.min_region_weight = float(min_region_weight)
        self.min_scribble_pixels = float(min_scribble_pixels)
        self.lambda_intra = float(lambda_intra)
        self.lambda_inter = float(lambda_inter)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_scribble_proto = float(lambda_scribble_proto)
        self.eps = float(eps)

        if adjacency_matrix is None:
            adj = torch.ones(self.num_classes, self.num_classes) - torch.eye(self.num_classes)
        else:
            adj = torch.tensor(adjacency_matrix, dtype=torch.float32)
            if adj.shape != (self.num_classes, self.num_classes):
                raise ValueError(
                    f"adjacency_matrix must have shape {(self.num_classes, self.num_classes)}, "
                    f"got {tuple(adj.shape)}."
                )
            adj = adj * (1.0 - torch.eye(self.num_classes))
        self.register_buffer("adjacency_matrix", adj.float(), persistent=False)

    def forward(
        self,
        probs: torch.Tensor,
        f_mnp: torch.Tensor,
        edge_map: torch.Tensor,
        scribble: torch.Tensor,
        from_logits: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if probs.ndim != 4:
            raise ValueError(f"Expected probs/logits [B, C, H, W], got {tuple(probs.shape)}.")
        if probs.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {probs.shape[1]}.")

        probs = torch.softmax(probs, dim=1) if from_logits else probs
        probs = probs.clamp_min(self.eps)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        f_mnp = _resize_like(f_mnp, probs)
        edge_map = _resize_like(edge_map, probs)
        if edge_map.shape[1] != 1:
            edge_map = edge_map.mean(dim=1, keepdim=True)
        edge_map = edge_map.clamp(0.0, 1.0)

        scribble_oh, scribble_valid_pixel = _one_hot_ignore(
            scribble, self.num_classes, self.ignore_index
        )
        if scribble_oh.shape[-2:] != probs.shape[-2:]:
            scribble_oh = _resize_like(scribble_oh, probs, mode="nearest")
            scribble_valid_pixel = scribble_oh.sum(dim=1) > 0

        features = F.normalize(f_mnp.float(), dim=1, eps=self.eps)
        class_weights = probs.pow(self.probability_power)

        proto, region_mass = self._class_prototypes(features, class_weights)
        scribble_proto, scribble_mass = self._class_prototypes(features, scribble_oh)

        valid_pred_class = region_mass > self.min_region_weight
        valid_scribble_class = scribble_mass > self.min_scribble_pixels
        valid_class_mask = valid_pred_class | valid_scribble_class

        loss_intra, intra_quality = self._intra_loss(
            features=features,
            proto=proto,
            weights=class_weights,
            valid_mask=valid_pred_class,
        )
        loss_inter, inter_quality = self._inter_loss(proto=proto, valid_mask=valid_pred_class)
        loss_boundary, boundary_quality, boundary_strength = self._boundary_loss(
            probs=probs,
            edge_map=edge_map,
        )
        loss_scr_proto, scribble_quality = self._scribble_proto_loss(
            proto=proto,
            scribble_proto=scribble_proto,
            valid_pred=valid_pred_class,
            valid_scribble=valid_scribble_class,
        )

        loss = (
            self.lambda_intra * loss_intra
            + self.lambda_inter * loss_inter
            + self.lambda_boundary * loss_boundary
            + self.lambda_scribble_proto * loss_scr_proto
        )

        class_quality = self._class_quality(
            intra_quality=intra_quality,
            inter_quality=inter_quality,
            scribble_quality=scribble_quality,
            boundary_quality=boundary_quality,
            valid_class_mask=valid_class_mask,
        )
        pixel_reliability = self._pixel_reliability(
            probs=probs,
            class_quality=class_quality,
            scribble_valid_pixel=scribble_valid_pixel,
        )

        info = {
            "loss_intra": loss_intra.detach(),
            "loss_inter": loss_inter.detach(),
            "loss_boundary": loss_boundary.detach(),
            "loss_scribble_proto": loss_scr_proto.detach(),
            "class_quality": class_quality.detach(),
            "pixel_reliability": pixel_reliability.detach(),
            "valid_class_mask": valid_class_mask.detach(),
            "valid_pred_class": valid_pred_class.detach(),
            "valid_scribble_class": valid_scribble_class.detach(),
            "region_mass": region_mass.detach(),
            "scribble_mass": scribble_mass.detach(),
            "boundary_quality": boundary_quality.detach(),
            "boundary_strength": boundary_strength.detach(),
        }
        return loss, info

    def _class_prototypes(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mass = weights.flatten(2).sum(dim=-1)
        proto = torch.einsum("bdhw,bchw->bcd", features, weights)
        proto = proto / mass.clamp_min(self.eps).unsqueeze(-1)
        proto = F.normalize(proto, dim=-1, eps=self.eps)
        return proto, mass

    def _intra_loss(
        self,
        features: torch.Tensor,
        proto: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = torch.einsum("bdhw,bcd->bchw", features, proto)
        dissim = 1.0 - sim
        mass = weights.flatten(2).sum(dim=-1).clamp_min(self.eps)
        per_class_loss = (dissim * weights).flatten(2).sum(dim=-1) / mass
        loss = _safe_mean(per_class_loss, valid_mask, eps=self.eps)

        intra_quality = (1.0 - 0.5 * per_class_loss).clamp(0.0, 1.0)
        intra_quality = torch.where(valid_mask, intra_quality, torch.zeros_like(intra_quality))
        return loss, intra_quality

    def _inter_loss(
        self,
        proto: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = torch.einsum("bcd,bkd->bck", proto, proto)
        distance = (1.0 - sim).clamp_min(0.0)
        pair_loss = F.relu(self.inter_margin - distance)

        valid_pair = valid_mask[:, :, None] & valid_mask[:, None, :]
        offdiag = ~torch.eye(self.num_classes, dtype=torch.bool, device=proto.device)
        pair_mask = valid_pair & offdiag[None]
        adj = self.adjacency_matrix.to(device=proto.device, dtype=proto.dtype)
        weighted_mask = pair_mask.to(proto.dtype) * adj[None]
        loss = (pair_loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(self.eps)

        sep_quality = (distance / max(self.inter_margin, self.eps)).clamp(0.0, 1.0)
        weighted_sep = sep_quality * weighted_mask
        denom = weighted_mask.sum(dim=-1).clamp_min(self.eps)
        inter_quality = weighted_sep.sum(dim=-1) / denom
        inter_quality = torch.where(valid_mask, inter_quality, torch.ones_like(inter_quality))
        return loss, inter_quality

    def _boundary_loss(
        self,
        probs: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_strength = _spatial_gradient_magnitude(probs).sum(dim=1, keepdim=True)
        boundary_strength = boundary_strength / boundary_strength.amax(
            dim=(2, 3), keepdim=True
        ).clamp_min(self.eps)

        numerator = (boundary_strength * (1.0 - edge_map)).sum(dim=(1, 2, 3))
        denom = boundary_strength.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        per_sample_loss = numerator / denom
        valid_boundary = denom > self.eps
        loss = _safe_mean(per_sample_loss, valid_boundary, eps=self.eps)
        boundary_quality = (1.0 - per_sample_loss).clamp(0.0, 1.0)
        return loss, boundary_quality, boundary_strength

    def _scribble_proto_loss(
        self,
        proto: torch.Tensor,
        scribble_proto: torch.Tensor,
        valid_pred: torch.Tensor,
        valid_scribble: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = (proto * scribble_proto).sum(dim=-1)
        valid = valid_pred & valid_scribble
        per_class_loss = 1.0 - sim
        loss = _safe_mean(per_class_loss, valid, eps=self.eps)

        quality = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        quality = torch.where(valid_scribble, quality, torch.ones_like(quality))
        quality = torch.where(valid_pred, quality, torch.zeros_like(quality))
        return loss, quality

    def _class_quality(
        self,
        intra_quality: torch.Tensor,
        inter_quality: torch.Tensor,
        scribble_quality: torch.Tensor,
        boundary_quality: torch.Tensor,
        valid_class_mask: torch.Tensor,
    ) -> torch.Tensor:
        boundary = boundary_quality[:, None].expand_as(intra_quality)
        class_quality = (
            0.45 * intra_quality
            + 0.25 * inter_quality
            + 0.20 * scribble_quality
            + 0.10 * boundary
        )
        return torch.where(valid_class_mask, class_quality.clamp(0.0, 1.0), torch.zeros_like(class_quality))

    def _pixel_reliability(
        self,
        probs: torch.Tensor,
        class_quality: torch.Tensor,
        scribble_valid_pixel: torch.Tensor,
    ) -> torch.Tensor:
        confidence, pred = probs.max(dim=1)
        gathered_quality = class_quality.gather(1, pred.flatten(1)).view_as(confidence)
        reliability = confidence * gathered_quality
        reliability = torch.where(scribble_valid_pixel, torch.ones_like(reliability), reliability)
        return reliability.clamp(0.0, 1.0)


__all__ = ["CMNPLoss"]
