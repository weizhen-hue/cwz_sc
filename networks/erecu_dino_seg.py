"""DINO-S/8 segmentation wrapper for EReCu-Med.

This module combines the DINO ViT backbone with lightweight mask heads. It is
intended to be instantiated twice in training: one student and one EMA teacher.
The model itself does not update EMA weights and does not build pseudo labels.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .dino_vit import VisionTransformer, build_dino_vits8, load_dino_vit_weights
from .heads import CoarseMaskHead, DSCHead, PatchSegHead, STAFHead


LayerSpec = Union[int, Sequence[int]]


def _normalize_layer_index(index: int, depth: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved = int(depth) + resolved
    if resolved < 0 or resolved >= int(depth):
        raise ValueError(f"Layer index {index} is out of range for depth {depth}.")
    return resolved


def _unique_sorted(indices: Iterable[int]) -> List[int]:
    return sorted(set(int(index) for index in indices))


class EReCuDinoSeg(nn.Module):
    """EReCu-style DINO-S/8 model with main, DSC, coarse, and STAF outputs."""

    def __init__(
        self,
        num_classes: int = 4,
        img_size: Union[int, Sequence[int]] = 256,
        input_channels: int = 1,
        dino_channels: int = 3,
        repeat_gray_to_rgb: bool = True,
        dino_pretrained_path: Optional[str] = None,
        dino_checkpoint_key: Optional[str] = None,
        main_layer: int = -1,
        dsc_layer: int = 3,
        coarse_layer: int = -1,
        staf_layers: Sequence[int] = (3, 7, 11),
        extra_layers: Optional[Sequence[int]] = None,
        head_hidden_channels: int = 192,
        main_hidden_channels: int = 256,
        freeze_backbone: bool = False,
        return_logits_only_by_default: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.dino_channels = int(dino_channels)
        self.repeat_gray_to_rgb = bool(repeat_gray_to_rgb)
        self.return_logits_only_by_default = bool(return_logits_only_by_default)

        self.backbone = build_dino_vits8(
            pretrained_path=dino_pretrained_path,
            checkpoint_key=dino_checkpoint_key,
            img_size=img_size,
            in_chans=self.dino_channels,
        )
        embed_dim = int(self.backbone.embed_dim)
        depth = int(self.backbone.depth)

        self.main_layer = _normalize_layer_index(main_layer, depth)
        self.dsc_layer = _normalize_layer_index(dsc_layer, depth)
        self.coarse_layer = _normalize_layer_index(coarse_layer, depth)
        self.staf_layers = tuple(_normalize_layer_index(index, depth) for index in staf_layers)
        self.extra_layers = tuple(
            _normalize_layer_index(index, depth) for index in (extra_layers or ())
        )
        self.selected_layers = _unique_sorted(
            [
                self.main_layer,
                self.dsc_layer,
                self.coarse_layer,
                *self.staf_layers,
                *self.extra_layers,
            ]
        )

        self.input_adapter = self._build_input_adapter()
        self.main_head = PatchSegHead(
            in_channels=embed_dim,
            num_classes=self.num_classes,
            hidden_channels=main_hidden_channels,
        )
        self.dsc_head = DSCHead(
            in_channels=embed_dim,
            num_classes=self.num_classes,
            hidden_channels=head_hidden_channels,
        )
        self.coarse_head = CoarseMaskHead(
            in_channels=embed_dim,
            num_classes=self.num_classes,
            hidden_channels=head_hidden_channels,
        )
        self.staf_head = STAFHead(
            in_channels=[embed_dim] * len(self.staf_layers),
            num_classes=self.num_classes,
            hidden_channels=head_hidden_channels,
            num_layers=len(self.staf_layers),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _build_input_adapter(self) -> nn.Module:
        if self.input_channels == self.dino_channels:
            return nn.Identity()
        if self.input_channels == 1 and self.dino_channels == 3 and self.repeat_gray_to_rgb:
            return GrayToRGB()
        return nn.Conv2d(self.input_channels, self.dino_channels, kernel_size=1, bias=True)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def load_backbone_weights(
        self,
        checkpoint_path: str,
        checkpoint_key: Optional[str] = None,
        strict: bool = False,
        map_location: str = "cpu",
    ) -> Dict[str, object]:
        return load_dino_vit_weights(
            self.backbone,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            strict=strict,
            map_location=map_location,
        )

    def _extract_layer_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        outputs = self.backbone.get_intermediate_layers(
            x,
            n=self.selected_layers,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        if len(outputs) != len(self.selected_layers):
            raise RuntimeError(
                f"Expected {len(self.selected_layers)} DINO layer outputs, got {len(outputs)}."
            )
        return {layer: feature for layer, feature in zip(self.selected_layers, outputs)}

    def forward(
        self,
        x: torch.Tensor,
        return_aux: Optional[bool] = None,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the model.

        Args:
            x: Input image tensor [B, input_channels, H, W].
            return_aux: If False, return only main logits. If True, return a
                dictionary with all EReCu auxiliary logits.
            return_features: Include selected DINO feature maps in the output.
            return_attention: Include last-layer self-attention.
        """
        if return_aux is None:
            return_aux = not self.return_logits_only_by_default

        output_size = x.shape[-2:]
        dino_input = self.input_adapter(x)
        layer_features = self._extract_layer_features(dino_input)

        main_feature = layer_features[self.main_layer]
        logits = self.main_head(main_feature, output_size=output_size)

        if not return_aux and not return_features and not return_attention:
            return logits

        dsc_feature = layer_features[self.dsc_layer]
        coarse_feature = layer_features[self.coarse_layer]
        staf_features = [layer_features[index] for index in self.staf_layers]

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "dsc_logits": self.dsc_head(dsc_feature, output_size=output_size),
            "coarse_logits": self.coarse_head(coarse_feature, output_size=output_size),
            "staf_logits": self.staf_head(staf_features, output_size=output_size),
        }

        if return_features:
            outputs["features"] = layer_features
            outputs["main_feature"] = main_feature
            outputs["dsc_feature"] = dsc_feature
            outputs["coarse_feature"] = coarse_feature

        if return_attention:
            outputs["last_selfattention"] = self.backbone.get_last_selfattention(dino_input)

        return outputs


class GrayToRGB(nn.Module):
    """Repeat a one-channel medical image to RGB for DINO checkpoint reuse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(f"GrayToRGB expects one input channel, got {x.shape[1]}.")
        return x.repeat(1, 3, 1, 1)


def build_erecu_dino_seg(
    num_classes: int = 4,
    img_size: Union[int, Sequence[int]] = 256,
    input_channels: int = 1,
    dino_pretrained_path: Optional[str] = None,
    dino_checkpoint_key: Optional[str] = None,
    **kwargs,
) -> EReCuDinoSeg:
    return EReCuDinoSeg(
        num_classes=num_classes,
        img_size=img_size,
        input_channels=input_channels,
        dino_pretrained_path=dino_pretrained_path,
        dino_checkpoint_key=dino_checkpoint_key,
        **kwargs,
    )


__all__ = [
    "EReCuDinoSeg",
    "GrayToRGB",
    "VisionTransformer",
    "build_erecu_dino_seg",
]
