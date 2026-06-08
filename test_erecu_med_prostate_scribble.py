"""Evaluate an EReCu-Med checkpoint on Prostate volumes.

The Prostate semi-supervised dataloader in this repository exposes train/val
only, although fold1 also defines patient001-patient015 as the held-out test
set. This script reads Prostate_training_volumes directly so that table-ready
test metrics can be computed without changing the training dataloader.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from EReCu.networks.erecu_dino_seg import build_erecu_dino_seg


DEFAULT_CLASS_NAMES = ("PZ", "CG")


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
        help="Which state dict to load when checkpoint contains multiple models.",
    )
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--dino_pretrained_path", type=str, default="")
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument(
        "--empty_hd95",
        type=float,
        default=50.0,
        help="HD95 fallback when exactly one of prediction/GT is empty.",
    )
    parser.add_argument(
        "--case_ids",
        type=str,
        default="",
        help="Optional comma-separated patient IDs or file stems, e.g. patient001,patient002.",
    )
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--save_summary_csv", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument(
        "--save_pred_dir",
        type=str,
        default="",
        help="Optional directory for saving predicted label volumes as .npy files.",
    )
    return parser.parse_args()


def setup_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _safe_torch_load(checkpoint_path: str, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _checkpoint_metadata(checkpoint: Dict[str, object], resolved_model_key: str) -> Dict[str, object]:
    skipped = {
        "student_state_dict",
        "teacher_state_dict",
        "model_state_dict",
        "state_dict",
        "optimizer_state_dict",
    }
    metadata = {key: _jsonable(value) for key, value in checkpoint.items() if key not in skipped}
    metadata["resolved_model_key"] = resolved_model_key
    return metadata


def select_state_dict(checkpoint, model_key: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    candidates = {
        "student": "student_state_dict",
        "teacher": "teacher_state_dict",
        "model": "model_state_dict",
        "state_dict": "state_dict",
    }

    if model_key != "auto":
        key = candidates[model_key]
        if model_key == "state_dict" and key not in checkpoint:
            return checkpoint, {"resolved_model_key": "raw_state_dict", "checkpoint_format": "raw_state_dict"}
        if key not in checkpoint:
            raise KeyError(f"Requested --model_key {model_key}, but checkpoint has no {key}.")
        return checkpoint[key], _checkpoint_metadata(checkpoint, resolved_model_key=model_key)

    for resolved_key, checkpoint_key in (
        ("student", "student_state_dict"),
        ("teacher", "teacher_state_dict"),
        ("model", "model_state_dict"),
        ("state_dict", "state_dict"),
    ):
        if checkpoint_key in checkpoint:
            return checkpoint[checkpoint_key], _checkpoint_metadata(checkpoint, resolved_model_key=resolved_key)

    return checkpoint, {"resolved_model_key": "raw_state_dict", "checkpoint_format": "raw_state_dict"}


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    model_key: str,
) -> Dict[str, object]:
    checkpoint = _safe_torch_load(checkpoint_path, device)
    state_dict, metadata = select_state_dict(checkpoint, model_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
        print(missing[:20])
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
        print(unexpected[:20])
    metadata["missing_keys"] = len(missing)
    metadata["unexpected_keys"] = len(unexpected)
    return metadata


def get_fold_patient_ids(fold: str, split: str) -> List[str]:
    if fold != "fold1":
        raise ValueError("Only fold1 is defined by prostate_dataset_semi.py in this repository.")
    if split == "test":
        ids = range(1, 16)
    elif split == "val":
        ids = range(16, 26)
    else:
        raise ValueError(f"Unsupported split: {split}")
    return [f"patient{idx:03d}" for idx in ids]


def _split_case_ids(case_ids: str) -> set[str]:
    return {item.strip() for item in case_ids.split(",") if item.strip()}


def _case_matches_filter(path: Path, filters: set[str]) -> bool:
    if not filters:
        return True
    stem = path.stem
    patient_id = stem.split("_")[0]
    return path.name in filters or stem in filters or patient_id in filters


def collect_volume_paths(root_path: str, fold: str, split: str, case_ids: str = "") -> List[Path]:
    volume_dir = Path(root_path) / "Prostate_training_volumes"
    if not volume_dir.exists():
        raise FileNotFoundError(f"Volume directory not found: {volume_dir}")

    patient_ids = get_fold_patient_ids(fold, split)
    filters = _split_case_ids(case_ids)
    paths: List[Path] = []
    for patient_id in patient_ids:
        matched = sorted(
            path
            for path in volume_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".h5", ".hdf5"}
            and re.match(rf"^{re.escape(patient_id)}($|[_\-.]).*", path.name) is not None
            and _case_matches_filter(path, filters)
        )
        paths.extend(matched)
    return paths


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray, empty_hd95: float) -> Tuple[float, float]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, float(empty_hd95)
    return float(metric.binary.dc(pred, gt)), float(metric.binary.hd95(pred, gt))


@torch.no_grad()
def infer_single_volume(
    image: np.ndarray,
    model: torch.nn.Module,
    patch_size: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    if image.ndim == 2:
        image_3d = image[None, ...]
        squeeze_output = True
    elif image.ndim == 3:
        image_3d = image
        squeeze_output = False
    else:
        raise ValueError(f"Expected 2D or 3D image volume, got shape {image.shape}.")

    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    prediction = np.zeros(image_3d.shape, dtype=np.uint8)
    model.eval()

    for slice_idx in range(image_3d.shape[0]):
        slice_raw = image_3d[slice_idx]
        h, w = slice_raw.shape
        slice_in = zoom(slice_raw, (patch_h / h, patch_w / w), order=0)
        input_tensor = torch.from_numpy(slice_in).unsqueeze(0).unsqueeze(0).float().to(device)

        outputs = model(input_tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
        pred_np = pred.cpu().numpy().astype(np.uint8)
        prediction[slice_idx] = zoom(pred_np, (h / patch_h, w / patch_w), order=0).astype(np.uint8)

    if squeeze_output:
        return prediction[0]
    return prediction


def class_names_for(num_classes: int) -> Tuple[str, ...]:
    foreground_classes = max(int(num_classes) - 1, 0)
    if foreground_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return tuple(f"class_{idx}" for idx in range(1, int(num_classes)))


def evaluate_case(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        image = handle["image"][:]
        label = handle["label"][:]

    prediction = infer_single_volume(image, model, args.patch_size, device)
    if args.save_pred_dir:
        os.makedirs(args.save_pred_dir, exist_ok=True)
        np.save(os.path.join(args.save_pred_dir, f"{path.stem}_pred.npy"), prediction)

    metric_list = []
    for class_idx in range(1, args.num_classes):
        metric_list.append(
            calculate_metric_percase(
                prediction == class_idx,
                label == class_idx,
                empty_hd95=args.empty_hd95,
            )
        )
    return path.stem, np.asarray(metric_list, dtype=np.float64)


def summarize_metrics(metrics: np.ndarray) -> Dict[str, object]:
    if metrics.ndim != 3 or metrics.shape[2] != 2:
        raise ValueError(f"Expected metrics [N, C-1, 2], got {metrics.shape}.")

    per_class_mean = metrics.mean(axis=0)
    per_class_std = metrics.std(axis=0, ddof=0)
    per_case_mean = metrics.mean(axis=1)
    mean_column_mean = per_case_mean.mean(axis=0)
    mean_column_std = per_case_mean.std(axis=0, ddof=0)
    return {
        "num_cases": int(metrics.shape[0]),
        "per_class_mean": per_class_mean.tolist(),
        "per_class_std": per_class_std.tolist(),
        "mean_column_mean": mean_column_mean.tolist(),
        "mean_column_std": mean_column_std.tolist(),
    }


def format_pm(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def write_case_csv(path: str, rows: List[List[object]], class_names: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["case"]
    for name in class_names:
        header.extend([f"{name}_dice", f"{name}_hd95"])
    header.extend(["Mean_dice", "Mean_hd95"])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_summary_csv(path: str, summary: Dict[str, object], class_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    per_class_mean = np.asarray(summary["per_class_mean"], dtype=np.float64)
    per_class_std = np.asarray(summary["per_class_std"], dtype=np.float64)
    mean_column_mean = np.asarray(summary["mean_column_mean"], dtype=np.float64)
    mean_column_std = np.asarray(summary["mean_column_std"], dtype=np.float64)

    rows = []
    for idx, name in enumerate(class_names):
        rows.append(
            [
                name,
                per_class_mean[idx, 0],
                per_class_std[idx, 0],
                per_class_mean[idx, 1],
                per_class_std[idx, 1],
            ]
        )
    rows.append(
        [
            "Mean",
            mean_column_mean[0],
            mean_column_std[0],
            mean_column_mean[1],
            mean_column_std[1],
        ]
    )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "dice_mean", "dice_std", "hd95_mean", "hd95_std"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.patch_size) != 2:
        raise ValueError(f"--patch_size expects two integers, got {args.patch_size}.")

    setup_reproducibility(args.seed, bool(args.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_erecu_dino_seg(
        num_classes=args.num_classes,
        img_size=tuple(args.patch_size),
        input_channels=1,
        dino_pretrained_path=args.dino_pretrained_path or None,
        dino_checkpoint_key=args.dino_checkpoint_key or None,
    ).to(device)
    checkpoint_metadata = load_checkpoint(model, args.checkpoint, device, args.model_key)
    model.eval()

    volume_paths = collect_volume_paths(args.root_path, args.fold, args.split, args.case_ids)
    if not volume_paths:
        raise RuntimeError(
            f"{args.split} set is empty. Check root_path={args.root_path}, "
            f"fold={args.fold}, and case_ids={args.case_ids!r}."
        )

    class_names = class_names_for(args.num_classes)
    print(f"Evaluating Prostate split={args.split}, fold={args.fold}, cases={len(volume_paths)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model key: {checkpoint_metadata.get('resolved_model_key', args.model_key)}")

    all_metrics = []
    rows = []
    for path in tqdm(volume_paths):
        case_id, metric_i = evaluate_case(path, model, args, device)
        all_metrics.append(metric_i)
        case_mean = metric_i.mean(axis=0)
        rows.append([case_id, *metric_i.reshape(-1).tolist(), case_mean[0], case_mean[1]])

        class_parts = [
            f"{name} dice={metric_i[idx, 0]:.4f}, hd95={metric_i[idx, 1]:.2f}"
            for idx, name in enumerate(class_names)
        ]
        print(f"{case_id}: {' | '.join(class_parts)} | Mean dice={case_mean[0]:.4f}, hd95={case_mean[1]:.2f}")

    metrics = np.stack(all_metrics, axis=0)
    summary = summarize_metrics(metrics)
    per_class_mean = np.asarray(summary["per_class_mean"], dtype=np.float64)
    per_class_std = np.asarray(summary["per_class_std"], dtype=np.float64)
    mean_column_mean = np.asarray(summary["mean_column_mean"], dtype=np.float64)
    mean_column_std = np.asarray(summary["mean_column_std"], dtype=np.float64)

    print("\nTable-ready volume-level mean +/- std")
    for idx, name in enumerate(class_names):
        print(
            f"{name}: Dice {format_pm(per_class_mean[idx, 0], per_class_std[idx, 0], 4)}, "
            f"HD95 {format_pm(per_class_mean[idx, 1], per_class_std[idx, 1], 2)}"
        )
    print(
        f"Mean: Dice {format_pm(mean_column_mean[0], mean_column_std[0], 4)}, "
        f"HD95 {format_pm(mean_column_mean[1], mean_column_std[1], 2)}"
    )

    result = {
        "args": vars(args),
        "checkpoint_metadata": checkpoint_metadata,
        "class_names": class_names,
        "cases": [path.name for path in volume_paths],
        "summary": summary,
    }

    if args.save_csv:
        write_case_csv(args.save_csv, rows, class_names)
        print(f"Saved per-case CSV to {args.save_csv}")
    if args.save_summary_csv:
        write_summary_csv(args.save_summary_csv, summary, class_names)
        print(f"Saved summary CSV to {args.save_summary_csv}")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, ensure_ascii=False, indent=2)
        print(f"Saved JSON summary to {args.save_json}")


if __name__ == "__main__":
    main()
