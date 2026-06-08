#!/usr/bin/env python
"""Diagnose whether CMNP native cues separate Prostate PZ/CG regions.

This script is for analysis only. It computes DoG/LBP and optional SPSD CMNP
features on selected Prostate slices, measures class prototype separability
using ground-truth masks, and optionally compares model predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion, zoom

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional diagnostic dependency
    matplotlib = None
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover - optional diagnostic dependency
    Image = None
    ImageDraw = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from EReCu.losses.cmnp_loss import CMNPLoss
from EReCu.modules.mnp_dog_lbp_torch import MNPDogLBPFeatureExtractor
from EReCu.modules.mnp_roughness_torch import MNPRoughnessFeatureExtractor
from EReCu.modules.mnp_spsd_torch import MNPSPSDFeatureExtractor
from EReCu.networks.erecu_dino_seg import build_erecu_dino_seg
from EReCu.test_erecu_med_prostate_scribble import load_checkpoint


CLASS_NAMES = ("BG", "PZ", "CG")
EVAL_CLASS_NAMES = ("PZ", "CG")
TEST_IDS = list(range(1, 16))
VAL_IDS = list(range(16, 26))
TRAIN_IDS = list(range(26, 81))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and quantify CMNP separability on Prostate."
    )
    parser.add_argument("--root_path", type=str, default="./data/Prostate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--case_ids", type=str, default="")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--slices_per_case", type=int, default=6)
    parser.add_argument(
        "--slice_strategy",
        type=str,
        default="foreground",
        choices=["foreground", "middle", "even"],
    )
    parser.add_argument("--patch_size", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--ignore_index", type=int, default=4)
    parser.add_argument(
        "--mnp_variant",
        type=str,
        default="both",
        choices=["dog_lbp", "spsd", "roughness", "both"],
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_key", type=str, default="auto")
    parser.add_argument("--dino_pretrained_path", type=str, default="")
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def patient_number(path: Path) -> int:
    match = re.search(r"patient(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def collect_cases(root_path: str, split: str, case_ids: str, max_cases: int) -> List[Path]:
    volume_dir = Path(root_path) / "Prostate_training_volumes"
    if not volume_dir.exists():
        raise FileNotFoundError(f"Missing Prostate volume directory: {volume_dir}")

    files = sorted(volume_dir.glob("patient*.h5"))
    if case_ids.strip():
        wanted = {item.strip() for item in case_ids.split(",") if item.strip()}
        normalized = set()
        for item in wanted:
            if item.isdigit():
                normalized.add(f"patient{int(item):03d}")
            else:
                normalized.add(item)
        cases = [p for p in files if p.stem in normalized]
    else:
        if split == "test":
            valid_ids = set(TEST_IDS)
        elif split == "val":
            valid_ids = set(VAL_IDS)
        elif split == "train":
            valid_ids = set(TRAIN_IDS)
        else:
            valid_ids = set(TEST_IDS + VAL_IDS + TRAIN_IDS)
        cases = [p for p in files if patient_number(p) in valid_ids]

    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError("No Prostate cases found for the requested split/case_ids.")
    return cases


def read_h5_case(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        image = np.asarray(handle["image"])
        label = np.asarray(handle["label"])
    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(f"Expected 3D image/label in {path}, got {image.shape}, {label.shape}")
    return image, label.astype(np.int64)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    image = np.clip(image, lo, hi)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    return image.astype(np.float32)


def resize_slice(image: np.ndarray, label: np.ndarray, patch_size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = int(patch_size[0]), int(patch_size[1])
    zoom_factors = (target_h / image.shape[0], target_w / image.shape[1])
    image_r = zoom(image, zoom_factors, order=1)
    label_r = zoom(label, zoom_factors, order=0)
    return image_r.astype(np.float32), label_r.astype(np.int64)


def select_slices(label: np.ndarray, slices_per_case: int, strategy: str) -> List[int]:
    depth = label.shape[0]
    foreground_slices = np.where(np.any(label > 0, axis=(1, 2)))[0]
    if foreground_slices.size == 0:
        return [depth // 2]

    if strategy == "middle":
        center = int(foreground_slices[len(foreground_slices) // 2])
        half = max(1, slices_per_case // 2)
        indices = list(range(max(0, center - half), min(depth, center + half + 1)))
    elif strategy == "even":
        indices = np.linspace(0, depth - 1, min(slices_per_case, depth)).round().astype(int).tolist()
    else:
        if foreground_slices.size <= slices_per_case:
            indices = foreground_slices.tolist()
        else:
            indices = np.linspace(
                int(foreground_slices[0]),
                int(foreground_slices[-1]),
                slices_per_case,
            ).round().astype(int).tolist()
    return sorted(set(int(i) for i in indices))


def make_extractors(device: torch.device, variant: str) -> Dict[str, torch.nn.Module]:
    extractors: Dict[str, torch.nn.Module] = {}
    if variant in ("dog_lbp", "both"):
        extractors["dog_lbp"] = MNPDogLBPFeatureExtractor().to(device).eval()
    if variant in ("spsd", "both"):
        extractors["spsd"] = MNPSPSDFeatureExtractor().to(device).eval()
    if variant in ("roughness", "both"):
        extractors["roughness"] = MNPRoughnessFeatureExtractor().to(device).eval()
    return extractors


def make_model(args: argparse.Namespace, device: torch.device) -> Optional[torch.nn.Module]:
    if not args.checkpoint:
        return None
    model = build_erecu_dino_seg(
        num_classes=args.num_classes,
        img_size=args.patch_size,
        dino_pretrained_path=args.dino_pretrained_path or None,
        dino_checkpoint_key=args.dino_checkpoint_key or None,
    )
    load_checkpoint(model, args.checkpoint, device, args.model_key)
    model.to(device)
    model.eval()
    return model


def one_hot_label(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    clipped = label.clamp(0, num_classes - 1)
    return F.one_hot(clipped, num_classes=num_classes).permute(0, 3, 1, 2).float()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    return float(values[mask].mean().detach().cpu().item())


def prototype_metrics(
    f_mnp: torch.Tensor,
    label: torch.Tensor,
    probs: torch.Tensor,
    edge_map: torch.Tensor,
    cmnp_info: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    del probs
    f_norm = F.normalize(f_mnp, dim=1)
    label_2d = label[0]
    pz_mask = label_2d == 1
    cg_mask = label_2d == 2

    metrics: Dict[str, float] = {
        "pz_pixels": float(pz_mask.sum().item()),
        "cg_pixels": float(cg_mask.sum().item()),
    }
    if pz_mask.sum().item() == 0 or cg_mask.sum().item() == 0:
        metrics.update(
            {
                "pz_cg_distance": float("nan"),
                "pz_self": float("nan"),
                "pz_to_cg": float("nan"),
                "cg_self": float("nan"),
                "cg_to_pz": float("nan"),
                "margin_pz": float("nan"),
                "margin_cg": float("nan"),
                "boundary_edge_gt": float("nan"),
            }
        )
        return metrics

    pz_proto = f_norm[:, :, pz_mask].mean(dim=-1)
    cg_proto = f_norm[:, :, cg_mask].mean(dim=-1)
    pz_proto = F.normalize(pz_proto, dim=1)
    cg_proto = F.normalize(cg_proto, dim=1)

    sim_pz = (f_norm * pz_proto[:, :, None, None]).sum(dim=1)[0]
    sim_cg = (f_norm * cg_proto[:, :, None, None]).sum(dim=1)[0]
    distance = 1.0 - F.cosine_similarity(pz_proto, cg_proto, dim=1)[0]

    pz_self = masked_mean(sim_pz, pz_mask)
    pz_to_cg = masked_mean(sim_cg, pz_mask)
    cg_self = masked_mean(sim_cg, cg_mask)
    cg_to_pz = masked_mean(sim_pz, cg_mask)

    fg = (label_2d > 0).detach().cpu().numpy()
    pz_np = pz_mask.detach().cpu().numpy()
    cg_np = cg_mask.detach().cpu().numpy()
    boundary_np = (
        binary_dilation(pz_np) ^ binary_erosion(pz_np)
    ) | (
        binary_dilation(cg_np) ^ binary_erosion(cg_np)
    )
    boundary_np = boundary_np & binary_dilation(fg)
    boundary = torch.from_numpy(boundary_np).to(label.device)
    boundary_edge = masked_mean(edge_map[0, 0], boundary)

    class_quality = cmnp_info.get("class_quality")
    if class_quality is not None:
        metrics["cmnp_quality_pz"] = float(class_quality[0, 1].detach().cpu().item())
        metrics["cmnp_quality_cg"] = float(class_quality[0, 2].detach().cpu().item())

    metrics.update(
        {
            "pz_cg_distance": float(distance.detach().cpu().item()),
            "pz_self": pz_self,
            "pz_to_cg": pz_to_cg,
            "cg_self": cg_self,
            "cg_to_pz": cg_to_pz,
            "margin_pz": pz_self - pz_to_cg,
            "margin_cg": cg_self - cg_to_pz,
            "boundary_edge_gt": boundary_edge,
        }
    )
    return metrics


def dice_score(pred: np.ndarray, label: np.ndarray, cls: int) -> float:
    pred_mask = pred == cls
    label_mask = label == cls
    denom = pred_mask.sum() + label_mask.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred_mask, label_mask).sum() / denom)


def normalize_to_01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(array[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(array[finite].min()), float(array[finite].max())
    return np.clip((array - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def colorize_label(label: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [0, 180, 255],
            [255, 70, 70],
        ],
        dtype=np.float32,
    ) / 255.0
    clipped = np.clip(label.astype(np.int64), 0, len(palette) - 1)
    return palette[clipped]


def overlay_label(image: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray = np.repeat(normalize_to_01(image)[..., None], 3, axis=-1)
    color = colorize_label(label)
    mask = label > 0
    out = gray.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color[mask]
    return np.clip(out, 0.0, 1.0)


def _to_uint8_rgb(panel: np.ndarray, cmap: Optional[str]) -> np.ndarray:
    panel = np.asarray(panel, dtype=np.float32)
    if panel.ndim == 3 and panel.shape[-1] == 3:
        rgb = np.clip(panel, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    values = normalize_to_01(panel)
    if cmap == "coolwarm":
        rgb = np.zeros((*values.shape, 3), dtype=np.float32)
        rgb[..., 0] = values
        rgb[..., 1] = 0.25 + 0.5 * (1.0 - np.abs(values - 0.5) * 2.0)
        rgb[..., 2] = 1.0 - values
    elif cmap == "magma":
        rgb = np.stack(
            [
                np.clip(1.6 * values, 0.0, 1.0),
                np.clip(values ** 1.7, 0.0, 1.0),
                np.clip(0.35 + 0.5 * values ** 0.7, 0.0, 1.0),
            ],
            axis=-1,
        )
    elif cmap == "viridis":
        rgb = np.stack(
            [
                0.25 + 0.65 * values,
                0.15 + 0.75 * np.sqrt(values),
                0.55 * (1.0 - values) + 0.2,
            ],
            axis=-1,
        )
    else:
        rgb = np.repeat(values[..., None], 3, axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_panel_pil(
    out_path: Path,
    panels: Sequence[Tuple[str, np.ndarray, Optional[str]]],
    title: str,
) -> None:
    if Image is None or ImageDraw is None:
        return
    tile_w, tile_h = 256, 256
    title_h, label_h = 32, 24
    canvas = Image.new("RGB", (tile_w * len(panels), title_h + label_h + tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:180], fill=(0, 0, 0))
    for idx, (name, panel, cmap) in enumerate(panels):
        rgb = _to_uint8_rgb(panel, cmap)
        tile = Image.fromarray(rgb).resize((tile_w, tile_h), resample=Image.BILINEAR)
        x0 = idx * tile_w
        draw.text((x0 + 8, title_h + 4), name, fill=(0, 0, 0))
        canvas.paste(tile, (x0, title_h + label_h))
    canvas.save(out_path)


def save_panel(
    out_path: Path,
    image: np.ndarray,
    label: np.ndarray,
    edge_map: np.ndarray,
    sim_pz: np.ndarray,
    sim_cg: np.ndarray,
    pred: Optional[np.ndarray],
    title: str,
) -> None:
    columns = 7 if pred is not None else 6
    panels = [
        ("Image", normalize_to_01(image), "gray"),
        ("GT PZ/CG", overlay_label(image, label), None),
        ("Edge", normalize_to_01(edge_map), "magma"),
        ("PZ sim", normalize_to_01(sim_pz), "viridis"),
        ("CG sim", normalize_to_01(sim_cg), "viridis"),
        ("PZ-CG margin", normalize_to_01(sim_pz - sim_cg), "coolwarm"),
    ]
    if pred is not None:
        panels.insert(2, ("Prediction", overlay_label(image, pred), None))

    if plt is None:
        _save_panel_pil(out_path, panels, title)
        return

    fig, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 3.4), dpi=140)

    for ax, (name, panel, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_similarity_maps(
    f_mnp: torch.Tensor,
    label: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    f_norm = F.normalize(f_mnp, dim=1)
    label_2d = label[0]
    pz_mask = label_2d == 1
    cg_mask = label_2d == 2
    if pz_mask.sum().item() == 0 or cg_mask.sum().item() == 0:
        shape = tuple(label_2d.shape)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    pz_proto = F.normalize(f_norm[:, :, pz_mask].mean(dim=-1), dim=1)
    cg_proto = F.normalize(f_norm[:, :, cg_mask].mean(dim=-1), dim=1)
    sim_pz = (f_norm * pz_proto[:, :, None, None]).sum(dim=1)[0]
    sim_cg = (f_norm * cg_proto[:, :, None, None]).sum(dim=1)[0]
    return sim_pz.detach().cpu().numpy(), sim_cg.detach().cpu().numpy()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = [
        "cmnp_loss",
        "cmnp_quality_pz",
        "cmnp_quality_cg",
        "pz_cg_distance",
        "margin_pz",
        "margin_cg",
        "boundary_edge_gt",
        "dice_pz",
        "dice_cg",
    ]
    variants = sorted({str(row["variant"]) for row in rows})
    summary: List[Dict[str, object]] = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        out: Dict[str, object] = {"variant": variant, "num_slices": len(subset)}
        for key in keys:
            values = []
            for row in subset:
                value = row.get(key)
                if value is None:
                    continue
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fvalue):
                    values.append(fvalue)
            if values:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        summary.append(out)
    return summary


def main() -> None:
    args = parse_args()
    patch_size = args.patch_size
    if len(patch_size) == 1:
        patch_size = [patch_size[0], patch_size[0]]

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = collect_cases(args.root_path, args.split, args.case_ids, args.max_cases)
    extractors = make_extractors(device, args.mnp_variant)
    cmnp_loss = CMNPLoss(num_classes=args.num_classes, ignore_index=args.ignore_index).to(device)
    cmnp_loss.eval()
    model = make_model(args, device)

    rows: List[Dict[str, object]] = []
    if _MATPLOTLIB_IMPORT_ERROR is not None:
        print(f"[WARN] matplotlib unavailable: {_MATPLOTLIB_IMPORT_ERROR}")
        if _PIL_IMPORT_ERROR is None:
            print("[INFO] using PIL fallback for PNG panels.")
        else:
            print(f"[WARN] PIL fallback unavailable: {_PIL_IMPORT_ERROR}")
    with torch.no_grad():
        for case_path in cases:
            image_vol, label_vol = read_h5_case(case_path)
            slice_indices = select_slices(label_vol, args.slices_per_case, args.slice_strategy)
            for slice_idx in slice_indices:
                image_slice = normalize_image(image_vol[slice_idx])
                label_slice = label_vol[slice_idx]
                image_r, label_r = resize_slice(image_slice, label_slice, patch_size)

                image_tensor = torch.from_numpy(image_r).to(device)[None, None]
                label_tensor = torch.from_numpy(label_r).to(device)[None]
                gt_probs = one_hot_label(label_tensor, args.num_classes)

                pred_np: Optional[np.ndarray] = None
                if model is not None:
                    output = model(image_tensor)
                    logits = output["logits"] if isinstance(output, dict) else output
                    pred_np = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int64)

                for variant_name, extractor in extractors.items():
                    f_mnp, edge_map = extractor(image_tensor)
                    loss, info = cmnp_loss(gt_probs, f_mnp, edge_map, label_tensor)
                    sim_pz, sim_cg = compute_similarity_maps(f_mnp, label_tensor)
                    metrics = prototype_metrics(f_mnp, label_tensor, gt_probs, edge_map, info)

                    row: Dict[str, object] = {
                        "case": case_path.stem,
                        "slice": int(slice_idx),
                        "variant": variant_name,
                        "cmnp_loss": float(loss.detach().cpu().item()),
                    }
                    row.update(metrics)
                    if pred_np is not None:
                        row["dice_pz"] = dice_score(pred_np, label_r, 1)
                        row["dice_cg"] = dice_score(pred_np, label_r, 2)
                    rows.append(row)

                    title = (
                        f"{case_path.stem} slice {slice_idx} | {variant_name} | "
                        f"PZ margin {row.get('margin_pz', float('nan')):.3f}, "
                        f"CG margin {row.get('margin_cg', float('nan')):.3f}"
                    )
                    panel_path = out_dir / f"{case_path.stem}_z{slice_idx:03d}_{variant_name}.png"
                    save_panel(
                        panel_path,
                        image_r,
                        label_r,
                        edge_map[0, 0].detach().cpu().numpy(),
                        sim_pz,
                        sim_cg,
                        pred_np,
                        title,
                    )

    summary = summarize_rows(rows)
    write_csv(out_dir / "cmnp_diagnostics.csv", rows)
    write_csv(out_dir / "cmnp_diagnostics_summary.csv", summary)
    with (out_dir / "cmnp_diagnostics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} slice-level rows to {out_dir / 'cmnp_diagnostics.csv'}")
    for item in summary:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
