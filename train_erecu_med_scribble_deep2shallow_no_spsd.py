"""No-SPSD ablation for EReCu-Med deep-to-shallow training.

This entry point reuses the full D2S training pipeline and changes only the
CMNP native feature extractor:

    full:    DoG + LBP + SPSD
    ablated: DoG + LBP

The boundary edge map remains the same as the full version, so this ablation
primarily tests whether SPSD helps or hurts class prototypes, CMNP quality, and
pseudo-label reliability.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import EReCu.train_erecu_med_scribble_deep2shallow as base
from EReCu.modules.mnp_dog_lbp_torch import MNPDogLBPFeatureExtractor


MNP_VARIANT = "dog_lbp_no_spsd"


def build_models_and_modules(args, device: torch.device):
    img_size = tuple(args.patch_size)
    d2s_pairs = getattr(args, "deep_to_shallow_pairs_resolved", tuple())
    d2s_layers = sorted({layer for pair in d2s_pairs for layer in pair})
    student = base.build_erecu_dino_seg(
        num_classes=args.num_classes,
        img_size=img_size,
        input_channels=1,
        dino_pretrained_path=args.dino_pretrained_path or None,
        dino_checkpoint_key=args.dino_checkpoint_key or None,
        freeze_backbone=bool(args.freeze_backbone),
        extra_layers=d2s_layers,
    ).to(device)
    teacher = base.create_ema_model(student, device=device)

    mnp_extractor = MNPDogLBPFeatureExtractor().to(device)
    cmnp_loss = base.CMNPLoss(num_classes=args.num_classes, ignore_index=args.ignore_index).to(device)
    pef_fusion = base.PEFFusion().to(device)
    return student, teacher, mnp_extractor, cmnp_loss, pef_fusion


_base_build_run_config = base.build_run_config


def build_run_config(args, snapshot_path: str, extra: Optional[Dict[str, object]] = None):
    runtime_extra: Dict[str, object] = {"mnp_variant": MNP_VARIANT, "mnp_channels": 2}
    if extra:
        runtime_extra.update(extra)

    config = _base_build_run_config(args, snapshot_path, extra=runtime_extra)
    config["paths"]["script"] = str(Path(__file__).resolve())
    config["paths"]["base_training_script"] = str(Path(base.__file__).resolve())
    config["method"] = {
        "mnp_variant": MNP_VARIANT,
        "native_features": ["LBP", "DoG_abs"],
        "removed_features": ["SPSD"],
        "edge_map": "0.5 * DoG_abs + 0.5 * gradient_magnitude",
    }
    return config


def main() -> None:
    args = base.parse_args()
    args.mnp_variant = MNP_VARIANT
    if not args.out_dir and args.exp == "ACDC_EReCu_Med_D2S":
        args.exp = "ACDC_EReCu_Med_D2S_NoSPSD"

    args.deep_to_shallow_pairs_resolved = base.parse_deep_to_shallow_pairs(
        args.deep_to_shallow_pairs
    )

    base.build_models_and_modules = build_models_and_modules
    base.build_run_config = build_run_config

    base.setup_seed(args)
    if args.dry_run:
        base.dry_run(args)
        return

    snapshot_path = base.make_snapshot_path(args)
    base.configure_logging(snapshot_path, args)
    base.copy_code_snapshot(snapshot_path)
    base.save_run_config(snapshot_path, args)
    base.train(args, snapshot_path)


if __name__ == "__main__":
    main()
