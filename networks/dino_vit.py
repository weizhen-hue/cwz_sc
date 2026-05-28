"""DINO-compatible Vision Transformer backbones.

This module implements the ViT-S/8 backbone used by facebookresearch/dino in a
small self-contained form. It intentionally contains only the backbone and
weight-loading utilities. Segmentation heads, PEF, CMNP, and training logic
belong in separate EReCu modules.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _to_2tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError(f"Expected a pair, got {value}.")
        return int(pair[0]), int(pair[1])
    return int(value), int(value)


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std)
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std).clamp_(-2.0 * std, 2.0 * std)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop residual paths per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        attn_out, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to non-overlapping patch embeddings."""

    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = 224,
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, height, width = x.shape
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            raise ValueError(
                "Input height and width must be divisible by the patch size. "
                f"Got {(height, width)} and patch size {self.patch_size}."
            )
        x = self.proj(x)
        grid_size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class VisionTransformer(nn.Module):
    """Vision Transformer backbone compatible with DINO ViT checkpoints."""

    def __init__(
        self,
        img_size: Union[int, Sequence[int]] = 224,
        patch_size: Union[int, Sequence[int]] = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = int(embed_dim)
        self.depth = int(depth)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self.patch_embed.patch_size

    def no_weight_decay(self) -> set:
        return {"pos_embed", "cls_token"}

    def interpolate_pos_encoding(
        self,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        num_patch_tokens = tokens.shape[1] - 1
        num_pos_tokens = self.pos_embed.shape[1] - 1
        target_grid_h, target_grid_w = int(grid_size[0]), int(grid_size[1])

        if num_patch_tokens == num_pos_tokens and self.patch_embed.grid_size == grid_size:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = tokens.shape[-1]
        source_grid = int(math.sqrt(num_pos_tokens))
        if source_grid * source_grid != num_pos_tokens:
            raise ValueError(
                "Only square source position embeddings are supported. "
                f"Got {num_pos_tokens} patch position tokens."
            )

        patch_pos_embed = patch_pos_embed.reshape(1, source_grid, source_grid, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_grid_h, target_grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        if patch_pos_embed.shape[1] != num_patch_tokens:
            raise RuntimeError(
                "Interpolated position embedding size does not match token size: "
                f"{patch_pos_embed.shape[1]} vs {num_patch_tokens}."
            )
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size = x.shape[0]
        x, grid_size = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, grid_size)
        return self.pos_drop(x), grid_size

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        return_grid_size: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        x, grid_size = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        features = x if return_all_tokens else x[:, 0]
        if return_grid_size:
            return features, grid_size
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x, return_all_tokens=False)
        return self.head(x)

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.prepare_tokens(x)
        for index, block in enumerate(self.blocks):
            if index < len(self.blocks) - 1:
                x = block(x)
            else:
                return block(x, return_attention=True)
        raise RuntimeError("VisionTransformer contains no transformer blocks.")

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> List[TensorOrTuple]:
        """Return intermediate block outputs.

        Args:
            x: Input tensor of shape [B, C, H, W].
            n: If int, return the last n block outputs. If a sequence, return
                outputs at those zero-based block indices. Negative indices are
                accepted.
            reshape: Convert patch tokens to [B, C, H_p, W_p].
            return_class_token: Return the CLS token separately.
            norm: Apply the final normalization to each selected output.
        """
        x, grid_size = self.prepare_tokens(x)
        selected_indices = self._resolve_block_indices(n)
        outputs: List[TensorOrTuple] = []

        for index, block in enumerate(self.blocks):
            x = block(x)
            if index not in selected_indices:
                continue
            out = self.norm(x) if norm else x
            outputs.append(self._format_layer_output(out, grid_size, reshape, return_class_token))
        return outputs

    def _resolve_block_indices(self, n: Union[int, Sequence[int]]) -> List[int]:
        if isinstance(n, int):
            if n <= 0 or n > len(self.blocks):
                raise ValueError(f"n must be in [1, {len(self.blocks)}], got {n}.")
            return list(range(len(self.blocks) - n, len(self.blocks)))

        indices: List[int] = []
        for index in n:
            resolved = int(index)
            if resolved < 0:
                resolved = len(self.blocks) + resolved
            if resolved < 0 or resolved >= len(self.blocks):
                raise ValueError(f"Block index {index} is out of range for depth {len(self.blocks)}.")
            indices.append(resolved)
        return indices

    def _format_layer_output(
        self,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
        reshape: bool,
        return_class_token: bool,
    ) -> TensorOrTuple:
        if reshape:
            patch_tokens = self.tokens_to_feature_map(tokens, grid_size, has_cls_token=True)
            if return_class_token:
                return patch_tokens, tokens[:, 0]
            return patch_tokens

        if return_class_token:
            return tokens[:, 1:], tokens[:, 0]
        return tokens

    @staticmethod
    def tokens_to_feature_map(
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
        has_cls_token: bool = True,
    ) -> torch.Tensor:
        if has_cls_token:
            tokens = tokens[:, 1:]
        batch_size, num_tokens, channels = tokens.shape
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
        if num_tokens != grid_h * grid_w:
            raise ValueError(
                f"Token count {num_tokens} cannot be reshaped to grid {(grid_h, grid_w)}."
            )
        return tokens.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)


def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """Build a DINO/DeiT-style ViT-S backbone."""
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def dino_vits8(**kwargs) -> VisionTransformer:
    """Build DINO ViT-S/8 without downloading weights."""
    return vit_small(patch_size=8, num_classes=0, **kwargs)


def _unwrap_checkpoint(
    checkpoint: Dict,
    checkpoint_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if checkpoint_key is not None:
        if checkpoint_key not in checkpoint:
            raise KeyError(f"Checkpoint key '{checkpoint_key}' not found.")
        checkpoint = checkpoint[checkpoint_key]
    elif isinstance(checkpoint, dict):
        for candidate_key in ("teacher", "student", "model", "state_dict", "backbone"):
            if candidate_key in checkpoint and isinstance(checkpoint[candidate_key], dict):
                checkpoint = checkpoint[candidate_key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dict or contain a state dict.")
    return checkpoint


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    prefixes = (
        "module.backbone.",
        "module.teacher.",
        "module.student.",
        "module.",
        "backbone.",
        "teacher.",
        "student.",
    )
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        clean_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        cleaned[clean_key] = value
    return cleaned


def _resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    target_shape: torch.Size,
    target_grid_size: Tuple[int, int],
) -> torch.Tensor:
    if pos_embed.shape == target_shape:
        return pos_embed

    if pos_embed.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Position embeddings must have shape [1, N + 1, C].")

    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    target_tokens = int(target_shape[1]) - 1
    target_dim = int(target_shape[2])
    target_h, target_w = int(target_grid_size[0]), int(target_grid_size[1])
    source_tokens = patch_pos.shape[1]
    source_grid = int(math.sqrt(source_tokens))

    if source_grid * source_grid != source_tokens:
        raise ValueError(f"Cannot resize non-square position embeddings with {source_tokens} tokens.")
    if target_h * target_w != target_tokens:
        raise ValueError(
            f"Target grid {(target_h, target_w)} does not match target tokens {target_tokens}."
        )
    if patch_pos.shape[-1] != target_dim:
        raise ValueError(f"Position embedding dim mismatch: {patch_pos.shape[-1]} vs {target_dim}.")

    patch_pos = patch_pos.reshape(1, source_grid, source_grid, target_dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_tokens, target_dim)
    return torch.cat((cls_pos, patch_pos), dim=1)


def load_dino_vit_weights(
    model: VisionTransformer,
    checkpoint_path: Union[str, os.PathLike],
    checkpoint_key: Optional[str] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, object]:
    """Load DINO ViT weights from a local checkpoint.

    The loader accepts backbone-only checkpoints and common full-checkpoint
    layouts containing keys such as "teacher", "student", "model", or
    "state_dict". Extra projection-head keys are skipped when strict is False.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _unwrap_checkpoint(checkpoint, checkpoint_key=checkpoint_key)
    state_dict = _clean_state_dict(state_dict)

    model_state = model.state_dict()
    loadable = OrderedDict()
    skipped = OrderedDict()

    for key, value in state_dict.items():
        if key not in model_state:
            skipped[key] = "unexpected"
            continue
        if key == "pos_embed" and value.shape != model_state[key].shape:
            value = _resize_abs_pos_embed(value, model_state[key].shape, model.patch_embed.grid_size)
        if value.shape != model_state[key].shape:
            skipped[key] = f"shape mismatch {tuple(value.shape)} != {tuple(model_state[key].shape)}"
            continue
        loadable[key] = value

    message = model.load_state_dict(loadable, strict=strict)
    return {
        "loaded_keys": len(loadable),
        "skipped_keys": skipped,
        "missing_keys": list(message.missing_keys),
        "unexpected_keys": list(message.unexpected_keys),
    }


def build_dino_vits8(
    pretrained_path: Optional[Union[str, os.PathLike]] = None,
    checkpoint_key: Optional[str] = None,
    img_size: Union[int, Sequence[int]] = 224,
    in_chans: int = 3,
    **kwargs,
) -> VisionTransformer:
    """Build DINO ViT-S/8 and optionally load local pretrained weights."""
    model = dino_vits8(img_size=img_size, in_chans=in_chans, **kwargs)
    if pretrained_path:
        load_dino_vit_weights(model, pretrained_path, checkpoint_key=checkpoint_key, strict=False)
    return model


__all__ = [
    "Attention",
    "Block",
    "DropPath",
    "Mlp",
    "PatchEmbed",
    "VisionTransformer",
    "build_dino_vits8",
    "dino_vits8",
    "drop_path",
    "load_dino_vit_weights",
    "vit_small",
]
