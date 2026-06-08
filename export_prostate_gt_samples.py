"""Export Prostate GT slices as PNG panels for quick visual inspection."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = np.asarray(
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
    parser.add_argument("--out_dir", type=str, default="./outputs/prostate_gt_samples")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--slices_per_case", type=int, default=4)
    parser.add_argument(
        "--slice_strategy",
        type=str,
        default="foreground",
        choices=("foreground", "middle", "even", "random"),
    )
    parser.add_argument(
        "--case_ids",
        type=str,
        default="",
        help="Optional comma-separated patient IDs or volume stems, e.g. patient001,patient002.",
    )
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--label_key", type=str, default="label")
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--draw_contours", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def split_patient_ids(split: str) -> List[str]:
    if split == "test":
        ids = range(1, 16)
    elif split == "val":
        ids = range(16, 26)
    elif split == "train":
        ids = range(26, 81)
    elif split == "all":
        ids = range(1, 81)
    else:
        raise ValueError(f"Unsupported split: {split}")
    return [f"patient{idx:03d}" for idx in ids]


def parse_case_ids(case_ids: str) -> set[str]:
    return {item.strip() for item in case_ids.split(",") if item.strip()}


def case_matches(path: Path, filters: set[str]) -> bool:
    if not filters:
        return True
    stem = path.stem
    patient = stem.split("_")[0]
    return path.name in filters or stem in filters or patient in filters


def collect_volume_paths(root_path: str, split: str, case_ids: str) -> List[Path]:
    volume_dir = Path(root_path) / "Prostate_training_volumes"
    if not volume_dir.exists():
        raise FileNotFoundError(f"Volume directory not found: {volume_dir}")

    filters = parse_case_ids(case_ids)
    paths: List[Path] = []
    for patient in split_patient_ids(split):
        matched = sorted(
            path
            for path in volume_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".h5", ".hdf5"}
            and re.match(rf"^{re.escape(patient)}($|[_\-.]).*", path.name) is not None
            and case_matches(path, filters)
        )
        paths.extend(matched)
    return paths


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


def colorize_label(label: np.ndarray) -> np.ndarray:
    clipped = np.clip(label.astype(np.int64), 0, len(PALETTE) - 1)
    return PALETTE[clipped]


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
    alpha: float,
    draw_contours: bool,
) -> np.ndarray:
    gray = normalize_to_01(image_2d)
    image_rgb = np.stack([gray, gray, gray], axis=-1)
    label_rgb = colorize_label(label_2d)
    mask = label_2d > 0

    out = image_rgb.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - alpha) * image_rgb[mask] + alpha * label_rgb[mask]
    if draw_contours:
        boundary = label_boundary(label_2d)
        out[boundary] = label_rgb[boundary]
    return np.clip(out, 0.0, 1.0)


def select_slice_indices(label_3d: np.ndarray, num_slices: int, strategy: str) -> List[int]:
    depth = int(label_3d.shape[0])
    num_slices = max(1, min(int(num_slices), depth))
    if strategy == "middle":
        center = depth // 2
        offsets = list(range(-(num_slices // 2), num_slices - num_slices // 2))
        return sorted({min(max(center + offset, 0), depth - 1) for offset in offsets})
    if strategy == "even":
        return sorted(np.linspace(0, depth - 1, num_slices, dtype=int).tolist())
    if strategy == "random":
        return sorted(random.sample(range(depth), num_slices))

    fg_counts = np.asarray([np.count_nonzero(label_3d[idx] > 0) for idx in range(depth)])
    if fg_counts.max() == 0:
        return sorted(np.linspace(0, depth - 1, num_slices, dtype=int).tolist())
    return sorted(np.argsort(fg_counts)[-num_slices:].astype(int).tolist())


def save_panel(
    image_2d: np.ndarray,
    label_2d: np.ndarray,
    out_path: str,
    case_name: str,
    slice_idx: int,
    alpha: float,
    draw_contours: bool,
    dpi: int,
) -> None:
    gray = normalize_to_01(image_2d)
    image_rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = overlay_label(image_2d, label_2d, alpha=alpha, draw_contours=draw_contours)
    mask_rgb = colorize_label(label_2d)

    panels = [("Image", image_rgb), ("GT overlay", overlay), ("GT mask", mask_rgb)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{case_name}  slice {slice_idx}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_selected_csv(path: str, rows: Iterable[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "slice", "png", "fg_pixels", "cg_pixels", "pz_pixels"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = collect_volume_paths(args.root_path, args.split, args.case_ids)
    if not paths:
        raise RuntimeError(
            f"No Prostate volumes matched root_path={args.root_path}, split={args.split}, case_ids={args.case_ids!r}."
        )
    paths = paths[: max(1, int(args.max_cases))]

    os.makedirs(args.out_dir, exist_ok=True)
    selected_rows = []
    print(f"Exporting Prostate GT samples: split={args.split}, cases={len(paths)}, out_dir={args.out_dir}")

    for path in paths:
        with h5py.File(path, "r") as handle:
            image = handle[args.image_key][:]
            label = handle[args.label_key][:].astype(np.uint8)
        if image.ndim != 3 or label.ndim != 3:
            raise ValueError(f"{path.name}: expected 3D image/label, got image={image.shape}, label={label.shape}")

        case_name = path.stem
        slice_indices = select_slice_indices(label, args.slices_per_case, args.slice_strategy)
        for slice_idx in slice_indices:
            label_2d = label[slice_idx]
            out_name = f"{case_name}_slice_{slice_idx:03d}.png"
            out_path = os.path.join(args.out_dir, out_name)
            save_panel(
                image[slice_idx],
                label_2d,
                out_path=out_path,
                case_name=case_name,
                slice_idx=int(slice_idx),
                alpha=args.overlay_alpha,
                draw_contours=bool(args.draw_contours),
                dpi=args.dpi,
            )
            selected_rows.append(
                [
                    case_name,
                    int(slice_idx),
                    out_name,
                    int(np.count_nonzero(label_2d > 0)),
                    int(np.count_nonzero(label_2d == 1)),
                    int(np.count_nonzero(label_2d == 2)),
                ]
            )
            print(f"saved {out_path}")

    csv_path = os.path.join(args.out_dir, "selected_gt_slices.csv")
    write_selected_csv(csv_path, selected_rows)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
