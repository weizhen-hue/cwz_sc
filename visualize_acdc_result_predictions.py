"""Run prediction visualizations for an EReCu-Med ACDC result directory.

This wrapper is intentionally thin. It fills in the paths for one experiment
directory, finds a checkpoint if possible, then delegates actual inference and
PNG generation to ``visualize_erecu_med_predictions.py``.

Example:
    python code/EReCu/visualize_acdc_result_predictions.py \
        --checkpoint model/EReCu_Med_ACDC_MAAGfold_d2s_k2_mid_ep100_cmnp100_no_spsd/erecu_med_best_model.pth
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = (
    REPO_ROOT
    / "model"
    / "EReCu_Med_ACDC_MAAGfold_d2s_k2_mid_ep100_cmnp100_no_spsd"
)
VIS_SCRIPT = Path(__file__).resolve().with_name("visualize_erecu_med_predictions.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ACDC image/GT/prediction/error panels for EReCu-Med."
    )
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pth. If omitted, the script searches --result_dir.",
    )
    parser.add_argument("--root_path", type=Path, default=REPO_ROOT / "data" / "ACDC")
    parser.add_argument("--fold", type=str, default="MAAGfold")
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--model_key", type=str, default="auto", choices=("auto", "student", "teacher", "state_dict"))
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--dino_pretrained_path", type=Path, default=None)
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--case_ids", type=str, default="")
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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the delegated command without running inference.",
    )
    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def checkpoint_score(path: Path) -> tuple:
    name = path.name.lower()
    dice_match = re.search(r"dice_([0-9]+(?:\.[0-9]+)?)", name)
    dice = float(dice_match.group(1)) if dice_match else -1.0
    priority = 0
    if name == "erecu_med_best_model.pth":
        priority = 4
    elif name == "best_model.pth":
        priority = 3
    elif "best" in name:
        priority = 2
    elif dice >= 0:
        priority = 1
    return (priority, dice, path.stat().st_mtime)


def find_checkpoint(result_dir: Path) -> Optional[Path]:
    if not result_dir.exists():
        return None
    candidates = [path for path in result_dir.rglob("*.pth") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_score)


def append_arg(cmd: List[str], name: str, value) -> None:
    cmd.extend([name, str(value)])


def format_command(cmd: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def main() -> None:
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else find_checkpoint(result_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else result_dir / "prediction_vis"
    root_path = resolve_path(args.root_path)
    dino_pretrained_path = resolve_path(args.dino_pretrained_path)

    if checkpoint is None:
        raise SystemExit(
            "No checkpoint found. Put a .pth file in the result directory or pass "
            "--checkpoint explicitly.\n"
            f"searched result_dir: {result_dir}"
        )
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if not root_path.exists():
        raise SystemExit(f"ACDC root_path does not exist: {root_path}")

    cmd = [
        sys.executable,
        str(VIS_SCRIPT),
        "--root_path",
        str(root_path),
        "--fold",
        args.fold,
        "--split",
        args.split,
        "--checkpoint",
        str(checkpoint),
        "--model_key",
        args.model_key,
        "--num_classes",
        str(args.num_classes),
        "--patch_size",
        str(args.patch_size[0]),
        str(args.patch_size[1]),
        "--out_dir",
        str(out_dir),
        "--case_ids",
        args.case_ids,
        "--max_cases",
        str(args.max_cases),
        "--slices_per_case",
        str(args.slices_per_case),
        "--slice_strategy",
        args.slice_strategy,
        "--overlay_alpha",
        str(args.overlay_alpha),
        "--draw_contours",
        str(args.draw_contours),
        "--save_individual",
        str(args.save_individual),
        "--save_summary_grid",
        str(args.save_summary_grid),
        "--summary_grid_name",
        args.summary_grid_name,
        "--dpi",
        str(args.dpi),
        "--seed",
        str(args.seed),
        "--deterministic",
        str(args.deterministic),
    ]
    if dino_pretrained_path:
        append_arg(cmd, "--dino_pretrained_path", dino_pretrained_path)
    if args.dino_checkpoint_key:
        append_arg(cmd, "--dino_checkpoint_key", args.dino_checkpoint_key)

    print(format_command(cmd))
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
