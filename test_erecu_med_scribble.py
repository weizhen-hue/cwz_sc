"""Evaluate an EReCu-Med checkpoint on ACDC volumes.

This script is intended for report/table generation. It evaluates a fixed
checkpoint on a fixed split, writes per-case metrics, and writes summary
statistics in a table-ready format.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from dataloaders.dataset_semi import BaseDataSets
from EReCu.networks.erecu_dino_seg import build_erecu_dino_seg
from val_2D import test_single_volume


CLASS_NAMES = ("RV", "MYO", "LV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/ACDC")
    parser.add_argument("--fold", type=str, default="MAAGfold")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_key",
        type=str,
        default="auto",
        choices=("auto", "student", "teacher", "state_dict"),
        help="Which state dict to load when checkpoint contains multiple models.",
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--dino_pretrained_path", type=str, default="")
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--save_summary_csv", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def setup_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def select_state_dict(checkpoint, model_key: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    metadata: Dict[str, object] = {}
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if model_key == "student":
        if "student_state_dict" not in checkpoint:
            raise KeyError("Requested --model_key student, but checkpoint has no student_state_dict.")
        metadata = _checkpoint_metadata(checkpoint, resolved_model_key="student")
        return checkpoint["student_state_dict"], metadata
    if model_key == "teacher":
        if "teacher_state_dict" not in checkpoint:
            raise KeyError("Requested --model_key teacher, but checkpoint has no teacher_state_dict.")
        metadata = _checkpoint_metadata(checkpoint, resolved_model_key="teacher")
        return checkpoint["teacher_state_dict"], metadata
    if model_key == "state_dict":
        return checkpoint, {"resolved_model_key": "state_dict", "checkpoint_format": "raw_state_dict"}

    if "student_state_dict" in checkpoint:
        metadata = _checkpoint_metadata(checkpoint, resolved_model_key="student")
        return checkpoint["student_state_dict"], metadata
    if "teacher_state_dict" in checkpoint:
        metadata = _checkpoint_metadata(checkpoint, resolved_model_key="teacher")
        return checkpoint["teacher_state_dict"], metadata

    return checkpoint, {"resolved_model_key": "state_dict", "checkpoint_format": "raw_state_dict"}


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
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
    metadata = {
        key: _jsonable(value)
        for key, value in checkpoint.items()
        if key not in {"student_state_dict", "teacher_state_dict", "optimizer_state_dict"}
    }
    metadata["resolved_model_key"] = resolved_model_key
    return metadata


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


def format_pm(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def summarize_metrics(metrics: np.ndarray) -> Dict[str, object]:
    """Summarize [N, 3, 2] volume metrics.

    For the Mean column, this uses each case's mean across RV/MYO/LV and then
    computes mean/std across cases. This matches a table-level mean distribution
    better than averaging per-class std values.
    """
    if metrics.ndim != 3 or metrics.shape[1:] != (3, 2):
        raise ValueError(f"Expected metrics [N, 3, 2], got {metrics.shape}.")

    per_class_mean = metrics.mean(axis=0)
    per_class_std = metrics.std(axis=0, ddof=0)
    per_case_mean = metrics.mean(axis=1)
    mean_of_mean = per_case_mean.mean(axis=0)
    std_of_mean = per_case_mean.std(axis=0, ddof=0)

    summary = {
        "num_cases": int(metrics.shape[0]),
        "per_class_mean": per_class_mean.tolist(),
        "per_class_std": per_class_std.tolist(),
        "mean_column_mean": mean_of_mean.tolist(),
        "mean_column_std": std_of_mean.tolist(),
    }
    return summary


def write_case_csv(path: str, rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "case",
        "RV_dice",
        "RV_hd95",
        "MYO_dice",
        "MYO_hd95",
        "LV_dice",
        "LV_hd95",
        "Mean_dice",
        "Mean_hd95",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_summary_csv(path: str, summary: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    per_class_mean = np.asarray(summary["per_class_mean"], dtype=np.float64)
    per_class_std = np.asarray(summary["per_class_std"], dtype=np.float64)
    mean_column_mean = np.asarray(summary["mean_column_mean"], dtype=np.float64)
    mean_column_std = np.asarray(summary["mean_column_std"], dtype=np.float64)

    rows = []
    for idx, name in enumerate(CLASS_NAMES):
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

    test_set = BaseDataSets(base_dir=args.root_path, fold=args.fold, split=args.split)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if len(test_set) == 0:
        raise RuntimeError(
            f"{args.split} set is empty. Check dataset_semi.py fold split support and root_path."
        )

    print(f"Evaluating split={args.split}, fold={args.fold}, cases={len(test_set)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model key: {checkpoint_metadata.get('resolved_model_key', args.model_key)}")

    all_metrics = []
    rows = []
    for sampled_batch in test_loader:
        case_id = sampled_batch.get("idx", ["unknown"])
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]

        metric_i = test_single_volume(
            sampled_batch["image"],
            sampled_batch["label"],
            model,
            classes=args.num_classes,
            patch_size=args.patch_size,
        )
        metric_i = np.asarray(metric_i, dtype=np.float64)
        all_metrics.append(metric_i)
        case_mean = metric_i.mean(axis=0)
        flat = metric_i.reshape(-1)
        rows.append([case_id, *flat.tolist(), case_mean[0], case_mean[1]])
        print(
            f"{case_id}: "
            f"RV dice={metric_i[0,0]:.4f}, hd95={metric_i[0,1]:.2f} | "
            f"MYO dice={metric_i[1,0]:.4f}, hd95={metric_i[1,1]:.2f} | "
            f"LV dice={metric_i[2,0]:.4f}, hd95={metric_i[2,1]:.2f} | "
            f"Mean dice={case_mean[0]:.4f}, hd95={case_mean[1]:.2f}"
        )

    metrics = np.stack(all_metrics, axis=0)
    summary = summarize_metrics(metrics)
    per_class_mean = np.asarray(summary["per_class_mean"], dtype=np.float64)
    per_class_std = np.asarray(summary["per_class_std"], dtype=np.float64)
    mean_column_mean = np.asarray(summary["mean_column_mean"], dtype=np.float64)
    mean_column_std = np.asarray(summary["mean_column_std"], dtype=np.float64)

    print("\nTable-ready volume-level mean +/- std")
    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"{name}: Dice {format_pm(per_class_mean[idx,0], per_class_std[idx,0], 4)}, "
            f"HD95 {format_pm(per_class_mean[idx,1], per_class_std[idx,1], 2)}"
        )
    print(
        f"Mean: Dice {format_pm(mean_column_mean[0], mean_column_std[0], 4)}, "
        f"HD95 {format_pm(mean_column_mean[1], mean_column_std[1], 2)}"
    )

    result = {
        "args": vars(args),
        "checkpoint_metadata": checkpoint_metadata,
        "class_names": CLASS_NAMES,
        "summary": summary,
    }

    if args.save_csv:
        write_case_csv(args.save_csv, rows)
        print(f"Saved per-case CSV to {args.save_csv}")
    if args.save_summary_csv:
        write_summary_csv(args.save_summary_csv, summary)
        print(f"Saved summary CSV to {args.save_summary_csv}")
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved JSON summary to {args.save_json}")


if __name__ == "__main__":
    main()
