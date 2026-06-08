"""Save paper-style EReCu-Med segmentation visualizations for Prostate.

This is the Prostate counterpart of visualize_erecu_med_predictions.py. It
reads Prostate_training_volumes directly so test patient001-patient015 can be
visualized without relying on the Prostate dataloader's missing test split.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from EReCu.networks.erecu_dino_seg import build_erecu_dino_seg
from EReCu.test_erecu_med_prostate_scribble import (
    collect_volume_paths,
    load_checkpoint,
    setup_reproducibility,
)


CLASS_NAMES = ("BG", "PZ", "CG")
DEFAULT_PALETTE = np.asarray(
    [
        [0.00, 0.00, 0.00],  # BG
        [0.00, 0.62, 1.00],  # PZ
        [1.00, 0.18, 0.18],  # CG
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/Prostate")
    parser.add_argument("--fold", type=str, default="fold1")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_key",
        type=str,
        default="auto",
        choices=("auto", "student", "teacher", "model", "state_dict"),
    )
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--dino_pretrained_path", type=str, default="")
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--case_ids",
        type=str,
        default="",
        help="Comma-separated patient IDs or volume stems. Empty means first --max_cases cases.",
    )
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--slices_per_case", type=int, default=2)
    parser.add_argument(
        "--slice_strategy",
        type=str,
        default="foreground",
        choices=("foreground", "middle", "even"),
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--draw_contours", type=int, default=1)
    parser.add_argument("--save_individual", type=int, default=1)
    parser.add_argument("--save_summary_grid", type=int, default=1)
    parser.add_argument("--summary_grid_name", type=str, default="summary_grid.png")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--deterministic", type=int, default=1)
    return parser.parse_args()


def normalize_to_01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(image[finite], [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(image[finite].min()), float(image[finite].max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def colorize_label(label: np.ndarray, num_classes: int) -> np.ndarray:
    palette = DEFAULT_PALETTE
    if num_classes > len(palette):
        rng = np.random.default_rng(17)
        extra = rng.uniform(0.15, 1.0, size=(num_classes - len(palette), 3)).astype(np.float32)
        palette = np.concatenate([palette, extra], axis=0)
    clipped = np.clip(label.astype(np.int64), 0, num_classes - 1)
    return palette[clipped]


def label_boundary(label: np.ndarray) -> np.ndarray:
    label = label.astype(np.int64)
    boundary = np.zeros(label.shape, dtype=bool)
    boundary[:-1, :] |= label[:-1, :] != label[1:, :]
    boundary[1:, :] |= label[:-1, :] != label[1:, :]
    boundary[:, :-1] |= label[:, :-1] != label[:, 1:]
    boundary[:, 1:] |= label[:, :-1] != label[:, 1:]
    return boundary & (label > 0)


def overlay_label(
    image_2d: np.ndarray,
    label_2d: np.ndarray,
    num_classes: int,
    alpha: float,
    draw_contours: bool,
) -> np.ndarray:
    image_norm = normalize_to_01(image_2d)
    image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
    label_rgb = colorize_label(label_2d, num_classes)
    mask = label_2d > 0

    out = image_rgb.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - alpha) * image_rgb[mask] + alpha * label_rgb[mask]
    if draw_contours:
        boundary = label_boundary(label_2d)
        out[boundary] = label_rgb[boundary]
    return np.clip(out, 0.0, 1.0)


def error_overlay(image_2d: np.ndarray, pred_2d: np.ndarray, label_2d: np.ndarray) -> np.ndarray:
    image_norm = normalize_to_01(image_2d)
    out = np.stack([image_norm, image_norm, image_norm], axis=-1)
    correct = (pred_2d == label_2d) & (label_2d > 0)
    false_positive = (pred_2d > 0) & (label_2d == 0)
    false_negative = (pred_2d == 0) & (label_2d > 0)
    wrong_class = (pred_2d > 0) & (label_2d > 0) & (pred_2d != label_2d)

    alpha = 0.65
    for mask, color in (
        (correct, np.asarray([0.20, 0.90, 0.25], dtype=np.float32)),
        (false_positive, np.asarray([1.00, 0.10, 0.10], dtype=np.float32)),
        (false_negative, np.asarray([0.10, 0.35, 1.00], dtype=np.float32)),
        (wrong_class, np.asarray([1.00, 0.00, 1.00], dtype=np.float32)),
    ):
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0.0, 1.0)


def _model_logits(output: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    image = np.asarray(image)
    squeeze_back = False
    if image.ndim == 2:
        image = image[None, ...]
        squeeze_back = True

    prediction = np.zeros(image.shape, dtype=np.uint8)
    for z in range(image.shape[0]):
        image_2d = image[z]
        height, width = image_2d.shape
        resized = zoom(
            image_2d,
            (float(patch_size[0]) / height, float(patch_size[1]) / width),
            order=0,
        )
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = _model_logits(model(tensor))
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
        pred_np = pred.cpu().numpy().astype(np.uint8)
        prediction[z] = zoom(
            pred_np,
            (float(height) / patch_size[0], float(width) / patch_size[1]),
            order=0,
        ).astype(np.uint8)

    if squeeze_back:
        return prediction[0]
    return prediction


def select_slice_indices(label_3d: np.ndarray, num_slices: int, strategy: str) -> List[int]:
    depth = int(label_3d.shape[0])
    num_slices = max(1, min(int(num_slices), depth))
    if strategy == "middle":
        center = depth // 2
        offsets = list(range(-(num_slices // 2), num_slices - num_slices // 2))
        return sorted({min(max(center + offset, 0), depth - 1) for offset in offsets})
    if strategy == "even":
        return sorted(np.linspace(0, depth - 1, num_slices, dtype=int).tolist())

    fg_counts = np.asarray([np.count_nonzero(label_3d[i] > 0) for i in range(depth)])
    if fg_counts.max() == 0:
        return sorted(np.linspace(0, depth - 1, num_slices, dtype=int).tolist())
    return sorted(np.argsort(fg_counts)[-num_slices:].astype(int).tolist())


def filter_cases(paths: Sequence[Path], case_ids: str, max_cases: int) -> List[Path]:
    if case_ids.strip():
        wanted = {item.strip() for item in case_ids.split(",") if item.strip()}
        selected = []
        for path in paths:
            stem = path.stem
            patient = stem.split("_")[0]
            if path.name in wanted or stem in wanted or patient in wanted:
                selected.append(path)
        return selected
    return list(paths[: max(1, int(max_cases))])


def build_panel_images(
    image_2d: np.ndarray,
    label_2d: np.ndarray,
    pred_2d: np.ndarray,
    num_classes: int,
    overlay_alpha: float,
    draw_contours: bool,
) -> List[Tuple[str, np.ndarray]]:
    image_norm = normalize_to_01(image_2d)
    image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
    return [
        ("Image", image_rgb),
        ("GT", overlay_label(image_2d, label_2d, num_classes, overlay_alpha, draw_contours)),
        ("Prediction", overlay_label(image_2d, pred_2d, num_classes, overlay_alpha, draw_contours)),
        ("Error", error_overlay(image_2d, pred_2d, label_2d)),
    ]


def save_panel(
    panels: List[Tuple[str, np.ndarray]],
    out_path: str,
    case_name: str,
    slice_idx: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.25))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{case_name}  slice {slice_idx}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_summary_grid(
    records: List[Dict[str, object]],
    out_path: str,
    num_classes: int,
    overlay_alpha: float,
    draw_contours: bool,
    dpi: int,
) -> None:
    if not records:
        return
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12.6, max(2.6, 2.55 * n_rows)))
    if n_rows == 1:
        axes = axes[None, :]

    for row, item in enumerate(records):
        panels = build_panel_images(
            item["image"],
            item["label"],
            item["pred"],
            num_classes=num_classes,
            overlay_alpha=overlay_alpha,
            draw_contours=draw_contours,
        )
        for col, (title, image) in enumerate(panels):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11)
        axes[row, 0].set_ylabel(
            f"{item['case']}:{int(item['slice'])}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_selected_csv(path: str, rows: Iterable[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "slice", "png"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.patch_size) != 2:
        raise ValueError(f"--patch_size expects two integers, got {args.patch_size}.")

    setup_reproducibility(args.seed, bool(args.deterministic))
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_erecu_dino_seg(
        num_classes=args.num_classes,
        img_size=tuple(args.patch_size),
        input_channels=1,
        dino_pretrained_path=args.dino_pretrained_path or None,
        dino_checkpoint_key=args.dino_checkpoint_key or None,
    ).to(device)
    metadata = load_checkpoint(model, args.checkpoint, device, args.model_key)
    model.eval()

    paths = collect_volume_paths(args.root_path, args.fold, args.split, args.case_ids)
    cases = filter_cases(paths, args.case_ids, args.max_cases)
    if not cases:
        raise RuntimeError(f"No Prostate volumes matched split={args.split}, case_ids={args.case_ids!r}.")

    os.makedirs(args.out_dir, exist_ok=True)
    selected_rows = []
    summary_records = []
    print(f"Visualizing Prostate split={args.split}, fold={args.fold}, cases={len(cases)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model key: {metadata.get('resolved_model_key', args.model_key)}")

    for path in cases:
        with h5py.File(path, "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:].astype(np.uint8)
        pred = predict_volume(model, image, args.patch_size, device)
        slice_indices = select_slice_indices(label, args.slices_per_case, args.slice_strategy)
        case_stem = path.stem

        for slice_idx in slice_indices:
            panels = build_panel_images(
                image[slice_idx],
                label[slice_idx],
                pred[slice_idx],
                num_classes=args.num_classes,
                overlay_alpha=args.overlay_alpha,
                draw_contours=bool(args.draw_contours),
            )
            out_name = f"{case_stem}_slice_{slice_idx:03d}.png"
            out_path = os.path.join(args.out_dir, out_name)
            if args.save_individual:
                save_panel(panels, out_path, case_stem, int(slice_idx), args.dpi)
            selected_rows.append([case_stem, int(slice_idx), out_name])
            summary_records.append(
                {
                    "case": case_stem,
                    "slice": int(slice_idx),
                    "image": image[slice_idx],
                    "label": label[slice_idx],
                    "pred": pred[slice_idx],
                }
            )
            print(f"saved {out_path}")

    if args.save_summary_grid:
        grid_path = os.path.join(args.out_dir, args.summary_grid_name)
        save_summary_grid(
            summary_records,
            grid_path,
            num_classes=args.num_classes,
            overlay_alpha=args.overlay_alpha,
            draw_contours=bool(args.draw_contours),
            dpi=args.dpi,
        )
        print(f"saved {grid_path}")

    selected_csv = os.path.join(args.out_dir, "selected_slices.csv")
    write_selected_csv(selected_csv, selected_rows)
    print(f"saved {selected_csv}")


if __name__ == "__main__":
    main()
