"""Train EReCu-Med on the Prostate scribble-supervised dataset.

This script adapts the ACDC EReCu-Med D2S training pipeline to Prostate:

    DINO Student + EMA Teacher + PEF + CMNP + Prostate scribble supervision

It reuses the deep-to-shallow training implementation and changes only the
dataset protocol/defaults and MNP feature variant. No LPR, ResNet18-MNP, SSN,
or superpixel module is imported here.

Default setting follows the latest ACDC observation: CMNP uses DoG+LBP without
SPSD. Set ``--mnp_variant spsd`` to run the full DoG+LBP+SPSD ablation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import EReCu.train_erecu_med_scribble_deep2shallow as base
from EReCu.modules.mnp_dog_lbp_torch import MNPDogLBPFeatureExtractor
from EReCu.modules.mnp_spsd_torch import MNPSPSDFeatureExtractor


PROSTATE_CLASS_NAMES = ("BG", "CG", "PZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./data/Prostate")
    parser.add_argument("--exp", type=str, default="Prostate_EReCu_Med_D2S")
    parser.add_argument("--fold", type=str, default="fold1")
    parser.add_argument("--sup_type", type=str, default="scribble")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--ignore_index", type=int, default=4)
    parser.add_argument("--labeled_num", type=int, default=5)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_unlabeled", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--vis_num", type=int, default=2)

    parser.add_argument("--dino_pretrained_path", type=str, default="")
    parser.add_argument("--dino_checkpoint_key", type=str, default="")
    parser.add_argument("--freeze_backbone", type=int, default=0)
    parser.add_argument(
        "--deep_to_shallow_pairs",
        type=str,
        default="5:7,7:9,9:11",
        help="Comma-separated zero-based DINO layer pairs, formatted as shallow:deep.",
    )
    parser.add_argument(
        "--mnp_variant",
        type=str,
        default="dog_lbp",
        choices=("dog_lbp", "spsd"),
        help="CMNP native features: dog_lbp removes SPSD; spsd uses DoG+LBP+SPSD.",
    )

    parser.add_argument("--lambda_epl", type=float, default=1.0)
    parser.add_argument("--lambda_deep_teacher", type=float, default=1.0)
    parser.add_argument("--lambda_student_coarse", type=float, default=0.5)
    parser.add_argument("--lambda_teacher_main", type=float, default=0.5)
    parser.add_argument("--lambda_staf", type=float, default=0.5)
    parser.add_argument("--lambda_pseudo", type=float, default=0.5)
    parser.add_argument("--lambda_cons", type=float, default=0.05)
    parser.add_argument("--lambda_cmnp", type=float, default=1.0)

    parser.add_argument("--pseudo_conf_threshold", type=float, default=0.65)
    parser.add_argument("--pseudo_reliability_threshold", type=float, default=0.35)
    parser.add_argument("--pseudo_class_quality_threshold", type=float, default=0.0)

    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_size", type=int, default=64)
    return parser.parse_args()


def build_loaders(args: argparse.Namespace):
    from dataloaders.prostate_dataset_semi import BaseDataSets, RandomGenerator

    train_transform = base.transforms.Compose([RandomGenerator(args.patch_size)])
    labeled_set = BaseDataSets(
        base_dir=args.root_path,
        labeled_num=args.labeled_num,
        labeled_type="labeled",
        sup_type=args.sup_type,
        fold=args.fold,
        split="train",
        transform=train_transform,
    )
    labeled_loader = DataLoader(
        labeled_set,
        batch_size=args.labeled_bs if args.use_unlabeled else args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=base.worker_init_fn,
    )

    unlabeled_loader = None
    if args.use_unlabeled:
        unlabeled_bs = max(1, args.batch_size - args.labeled_bs)
        unlabeled_set = BaseDataSets(
            base_dir=args.root_path,
            labeled_num=args.labeled_num,
            labeled_type="unlabeled",
            sup_type=args.sup_type,
            fold=args.fold,
            split="train",
            transform=train_transform,
        )
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=unlabeled_bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=base.worker_init_fn,
        )

    val_set = BaseDataSets(
        base_dir=args.root_path,
        labeled_num=args.labeled_num,
        fold=args.fold,
        split="val",
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    base.logging.info("Prostate labeled slices: %d", len(labeled_set))
    if unlabeled_loader is not None:
        base.logging.info("Prostate unlabeled slices: %d", len(unlabeled_loader.dataset))
    base.logging.info("Prostate validation volumes: %d", len(val_set))
    return labeled_loader, unlabeled_loader, val_loader


def build_models_and_modules(args: argparse.Namespace, device: torch.device):
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

    if args.mnp_variant == "spsd":
        mnp_extractor = MNPSPSDFeatureExtractor().to(device)
        mnp_channels = mnp_extractor.out_channels
        native_features = ["LBP", "DoG_abs", "SPSD"]
    else:
        mnp_extractor = MNPDogLBPFeatureExtractor().to(device)
        mnp_channels = mnp_extractor.out_channels
        native_features = ["LBP", "DoG_abs"]

    cmnp_loss = base.CMNPLoss(num_classes=args.num_classes, ignore_index=args.ignore_index).to(device)
    pef_fusion = base.PEFFusion().to(device)
    args.mnp_channels = int(mnp_channels)
    args.native_features = native_features
    return student, teacher, mnp_extractor, cmnp_loss, pef_fusion


_base_build_run_config = base.build_run_config


def build_run_config(
    args: argparse.Namespace,
    snapshot_path: str,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    runtime_extra: Dict[str, object] = {
        "dataset": "Prostate",
        "class_names": PROSTATE_CLASS_NAMES,
        "labeled_num": int(args.labeled_num),
        "mnp_variant": args.mnp_variant,
        "mnp_channels": int(getattr(args, "mnp_channels", 0)),
        "native_features": getattr(args, "native_features", []),
        "split_protocol": "fold1: train patients 026-080, val 016-025, test 001-015; labeled train cases are first labeled_num training patients.",
    }
    if extra:
        runtime_extra.update(extra)

    config = _base_build_run_config(args, snapshot_path, extra=runtime_extra)
    config["paths"]["script"] = str(Path(__file__).resolve())
    config["paths"]["base_training_script"] = str(Path(base.__file__).resolve())
    config["method"] = {
        "framework": "EReCu-Med D2S",
        "dataset": "Prostate",
        "classes": PROSTATE_CLASS_NAMES,
        "mnp_variant": args.mnp_variant,
        "native_features": getattr(args, "native_features", []),
        "removed_modules": ["LPR/TAS/LPG", "ResNet18-MNP", "SSN/superpixel"],
    }
    return config


def main() -> None:
    args = parse_args()
    args.deep_to_shallow_pairs_resolved = base.parse_deep_to_shallow_pairs(
        args.deep_to_shallow_pairs
    )

    base.build_loaders = build_loaders
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
