"""Train EReCu-Med with explicit deep-to-shallow teacher-student supervision.

This script keeps the EReCu-style one-stage teacher-student training flow:
DINO Student + EMA Teacher + PEF + CMNP quality. It does not use LPR, ResNet18
MNP, SSN, or superpixels.

Compared with train_erecu_med_scribble.py, this variant makes the EReCu Fig. 4
logic more explicit: Teacher deep/coarse prediction and Student coarse
prediction supervise the Student shallow DSC branch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    from torchvision import transforms
except ImportError:
    class _Compose:
        def __init__(self, transforms_list):
            self.transforms_list = transforms_list

        def __call__(self, sample):
            for transform in self.transforms_list:
                sample = transform(sample)
            return sample

    class transforms:
        Compose = _Compose


CURRENT_DIR = Path(__file__).resolve().parent
CODE_DIR = CURRENT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from EReCu.losses.cmnp_loss import CMNPLoss
from EReCu.losses.segmentation_losses import (
    consistency_loss,
    partial_ce_dice_loss,
    soft_dice_loss,
    weighted_ce_dice_loss,
)
from EReCu.modules.ema import create_ema_model, set_ema_eval, update_ema_model
from EReCu.modules.mnp_dog_lbp_torch import MNPDogLBPFeatureExtractor
from EReCu.modules.mnp_roughness_torch import MNPRoughnessFeatureExtractor
from EReCu.modules.mnp_spsd_torch import MNPSPSDFeatureExtractor
from EReCu.modules.pef_med import PEFFusion
from EReCu.modules.pseudo_label import build_evolved_pseudo_label, pseudo_label_stats
from EReCu.networks.erecu_dino_seg import build_erecu_dino_seg


DINO_VITS_DEPTH = 12


def normalize_dino_layer(index: int, depth: int = DINO_VITS_DEPTH) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved = int(depth) + resolved
    if resolved < 0 or resolved >= int(depth):
        raise ValueError(f"DINO layer index {index} is out of range for depth {depth}.")
    return resolved


def parse_deep_to_shallow_pairs(pair_text: str, depth: int = DINO_VITS_DEPTH) -> Tuple[Tuple[int, int], ...]:
    """Parse shallow:deep pairs, e.g. '3:7,7:11'.

    Layer indices are zero-based DINO block indices. Negative indices follow
    Python convention, so -1 means the last block.
    """
    if not pair_text.strip():
        return tuple()

    pairs = []
    for raw_pair in pair_text.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        if ":" not in raw_pair:
            raise ValueError(
                f"Invalid deep_to_shallow pair '{raw_pair}'. Expected format shallow:deep."
            )
        shallow_text, deep_text = raw_pair.split(":", 1)
        shallow = normalize_dino_layer(int(shallow_text), depth=depth)
        deep = normalize_dino_layer(int(deep_text), depth=depth)
        if shallow >= deep:
            raise ValueError(
                f"Invalid deep_to_shallow pair '{raw_pair}'. "
                "The shallow layer must be earlier than the deep layer."
            )
        pairs.append((shallow, deep))

    return tuple(dict.fromkeys(pairs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="../data/ACDC")
    parser.add_argument("--exp", type=str, default="ACDC_EReCu_Med_D2S")
    parser.add_argument("--fold", type=str, default="MAAGfold")
    parser.add_argument("--sup_type", type=str, default="scribble")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--ignore_index", type=int, default=4)
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
        default="spsd",
        choices=("spsd", "dog_lbp", "roughness"),
        help=(
            "CMNP native features: spsd uses DoG+LBP+SPSD; "
            "dog_lbp removes SPSD; roughness replaces SPSD with local roughness."
        ),
    )
    parser.add_argument(
        "--roughness_weight",
        type=float,
        default=0.35,
        help="Scale applied to roughness channels before concatenating them into F_mnp.",
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
    parser.add_argument("--dry_run_size", type=int, default=32)
    return parser.parse_args()


def resolve_mnp_metadata(mnp_variant: str) -> Tuple[int, Tuple[str, ...]]:
    if mnp_variant == "roughness":
        return 9, (
            "LBP",
            "DoG_abs",
            "local_std_s1",
            "local_std_s2",
            "local_std_s4",
            "residual_s1",
            "residual_s2",
            "residual_s4",
            "laplacian_abs",
        )
    if mnp_variant == "dog_lbp":
        return 2, ("LBP", "DoG_abs")
    if mnp_variant == "spsd":
        return 18, ("LBP", "DoG_abs", "SPSD")
    raise ValueError(f"Unsupported mnp_variant: {mnp_variant}")


def setup_seed(args: argparse.Namespace) -> None:
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_snapshot_path(args: argparse.Namespace) -> str:
    if args.out_dir:
        snapshot_path = args.out_dir
    else:
        snapshot_path = os.path.join("..", "model", f"{args.exp}_{args.fold}", args.sup_type)
    os.makedirs(snapshot_path, exist_ok=True)
    return snapshot_path


def configure_logging(snapshot_path: str, args: argparse.Namespace) -> None:
    log_path = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_run_config(
    args: argparse.Namespace,
    snapshot_path: str,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "argv": sys.argv,
            "text": " ".join(sys.argv),
            "python_executable": sys.executable,
        },
        "paths": {
            "cwd": os.getcwd(),
            "script": str(Path(__file__).resolve()),
            "code_dir": str(CODE_DIR.resolve()),
            "snapshot_path": str(Path(snapshot_path).resolve()),
            "root_path": args.root_path,
            "root_path_resolved": str(Path(args.root_path).resolve()),
            "dino_pretrained_path": args.dino_pretrained_path,
            "dino_pretrained_path_resolved": (
                str(Path(args.dino_pretrained_path).resolve()) if args.dino_pretrained_path else ""
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "args": _json_ready(vars(args)),
    }
    if extra:
        config["runtime"] = _json_ready(extra)
    return config


def save_run_config(
    snapshot_path: str,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
    filename: str = "run_config.json",
) -> None:
    config_path = os.path.join(snapshot_path, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(build_run_config(args, snapshot_path, extra=extra), f, indent=2, sort_keys=True)
        f.write("\n")
    logging.info("saved run config to %s", config_path)


def save_checkpoint_metadata(
    path: str,
    args: argparse.Namespace,
    iter_num: int,
    performance: float,
    mean_hd95: float,
    selection_metric: str,
) -> None:
    metadata = {
        "checkpoint": os.path.basename(path),
        "iter_num": int(iter_num),
        "selection_metric": selection_metric,
        "mean_dice": float(performance),
        "mean_hd95": float(mean_hd95),
        "args": _json_ready(vars(args)),
    }
    meta_path = f"{os.path.splitext(path)[0]}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    logging.info("saved checkpoint metadata to %s", meta_path)


def copy_code_snapshot(snapshot_path: str) -> None:
    code_dst = os.path.join(snapshot_path, "code")
    if os.path.exists(code_dst):
        shutil.rmtree(code_dst)
    shutil.copytree(str(CODE_DIR), code_dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    from dataloaders.dataset_semi import BaseDataSets, RandomGenerator

    train_transform = transforms.Compose([RandomGenerator(args.patch_size)])
    labeled_set = BaseDataSets(
        base_dir=args.root_path,
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
        worker_init_fn=worker_init_fn,
    )

    unlabeled_loader = None
    if args.use_unlabeled:
        unlabeled_bs = max(1, args.batch_size - args.labeled_bs)
        unlabeled_set = BaseDataSets(
            base_dir=args.root_path,
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
            worker_init_fn=worker_init_fn,
        )

    val_set = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    logging.info("labeled slices: %d", len(labeled_set))
    if unlabeled_loader is not None:
        logging.info("unlabeled slices: %d", len(unlabeled_loader.dataset))
    logging.info("validation volumes: %d", len(val_set))
    return labeled_loader, unlabeled_loader, val_loader


def build_models_and_modules(args: argparse.Namespace, device: torch.device):
    img_size = tuple(args.patch_size)
    d2s_pairs = getattr(args, "deep_to_shallow_pairs_resolved", tuple())
    d2s_layers = sorted({layer for pair in d2s_pairs for layer in pair})
    student = build_erecu_dino_seg(
        num_classes=args.num_classes,
        img_size=img_size,
        input_channels=1,
        dino_pretrained_path=args.dino_pretrained_path or None,
        dino_checkpoint_key=args.dino_checkpoint_key or None,
        freeze_backbone=bool(args.freeze_backbone),
        extra_layers=d2s_layers,
    ).to(device)
    teacher = create_ema_model(student, device=device)

    if args.mnp_variant == "roughness":
        mnp_extractor = MNPRoughnessFeatureExtractor(
            roughness_weight=args.roughness_weight
        ).to(device)
    elif args.mnp_variant == "dog_lbp":
        mnp_extractor = MNPDogLBPFeatureExtractor().to(device)
    else:
        mnp_extractor = MNPSPSDFeatureExtractor().to(device)
    args.mnp_channels = int(mnp_extractor.out_channels)
    args.native_features = list(resolve_mnp_metadata(args.mnp_variant)[1])
    cmnp_loss = CMNPLoss(num_classes=args.num_classes, ignore_index=args.ignore_index).to(device)
    pef_fusion = PEFFusion().to(device)
    return student, teacher, mnp_extractor, cmnp_loss, pef_fusion


def prepare_batch(
    labeled_batch: Dict[str, torch.Tensor],
    unlabeled_batch: Optional[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_l = labeled_batch["image"].to(device, non_blocking=True).float()
    scribble_l = labeled_batch["label"].to(device, non_blocking=True).long()

    if unlabeled_batch is None:
        return image_l, scribble_l

    image_u = unlabeled_batch["image"].to(device, non_blocking=True).float()
    scribble_u = torch.full(
        (image_u.shape[0], image_u.shape[-2], image_u.shape[-1]),
        int(args.ignore_index),
        device=device,
        dtype=torch.long,
    )
    image = torch.cat([image_l, image_u], dim=0)
    scribble = torch.cat([scribble_l, scribble_u], dim=0)
    return image, scribble


def compute_train_step(
    images: torch.Tensor,
    scribble: torch.Tensor,
    student,
    teacher,
    mnp_extractor,
    cmnp_loss_fn,
    pef_fusion,
    args: argparse.Namespace,
    return_visuals: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    f_mnp, edge_map = mnp_extractor(images)

    student_out = student(images, return_aux=True, return_features=True)
    with torch.no_grad():
        set_ema_eval(teacher)
        teacher_out = teacher(images, return_aux=True, return_features=True)
        teacher_logits = teacher_out["logits"]
        teacher_coarse_logits = teacher_out["coarse_logits"]
        teacher_staf_logits = teacher_out["staf_logits"]

    loss_cmnp, cmnp_info = cmnp_loss_fn(
        student_out["dsc_logits"],
        f_mnp=f_mnp,
        edge_map=edge_map,
        scribble=scribble,
        from_logits=True,
    )

    p_evo, pef_info = pef_fusion(
        student_logits=student_out["logits"],
        teacher_logits=teacher_coarse_logits,
        dsc_logits=student_out["dsc_logits"],
        staf_logits=student_out["staf_logits"],
        cmnp_info=cmnp_info,
        inputs_are_logits=True,
    )

    pseudo_label, pseudo_weight = build_evolved_pseudo_label(
        probs_evo=p_evo,
        scribble=scribble,
        cmnp_info=cmnp_info,
        ignore_index=args.ignore_index,
        from_logits=False,
        confidence_threshold=args.pseudo_conf_threshold,
        reliability_threshold=args.pseudo_reliability_threshold,
        class_quality_threshold=args.pseudo_class_quality_threshold,
    )

    loss_scr = partial_ce_dice_loss(
        student_out["logits"],
        scribble,
        ignore_index=args.ignore_index,
    )

    d2s_pairs = getattr(args, "deep_to_shallow_pairs_resolved", tuple())
    output_size = images.shape[-2:]
    student_self_losses: Dict[int, torch.Tensor] = {}
    student_shallow_logits_cache: Dict[int, torch.Tensor] = {}
    teacher_pair_losses = []
    pair_metrics: Dict[str, torch.Tensor] = {}
    for shallow_layer, deep_layer in d2s_pairs:
        if shallow_layer not in student_shallow_logits_cache:
            student_shallow_logits_cache[shallow_layer] = student.dsc_head(
                student_out["features"][shallow_layer],
                output_size=output_size,
            )
        student_shallow_logits = student_shallow_logits_cache[shallow_layer]

        if shallow_layer not in student_self_losses:
            student_shallow_pool_logits = student.coarse_head(
                student_out["features"][shallow_layer],
                output_size=output_size,
            )
            loss_self = soft_dice_loss(
                student_shallow_logits,
                student_shallow_pool_logits.detach(),
                input_from_logits=True,
                target_from_logits=True,
            )
            student_self_losses[shallow_layer] = loss_self
            pair_metrics[f"loss_d2s_student_self_{shallow_layer}"] = loss_self.detach()

        teacher_deep_logits = teacher.coarse_head(
            teacher_out["features"][deep_layer],
            output_size=output_size,
        )

        loss_pair_teacher = soft_dice_loss(
            student_shallow_logits,
            teacher_deep_logits.detach(),
            input_from_logits=True,
            target_from_logits=True,
        )
        teacher_pair_losses.append(loss_pair_teacher)
        pair_metrics[f"loss_d2s_teacher_{shallow_layer}_{deep_layer}"] = loss_pair_teacher.detach()

    if student_self_losses:
        loss_epl_student_coarse = torch.stack(list(student_self_losses.values())).mean()
        loss_epl_teacher_coarse = torch.stack(teacher_pair_losses).mean()
    else:
        loss_epl_student_coarse = soft_dice_loss(
            student_out["dsc_logits"],
            student_out["coarse_logits"].detach(),
            input_from_logits=True,
            target_from_logits=True,
        )
        loss_epl_teacher_coarse = soft_dice_loss(
            student_out["dsc_logits"],
            teacher_coarse_logits.detach(),
            input_from_logits=True,
            target_from_logits=True,
        )

    loss_epl_teacher_main = soft_dice_loss(
        student_out["dsc_logits"],
        teacher_logits.detach(),
        input_from_logits=True,
        target_from_logits=True,
    )
    loss_epl = (
        args.lambda_student_coarse * loss_epl_student_coarse
        + args.lambda_deep_teacher * loss_epl_teacher_coarse
        + args.lambda_teacher_main * loss_epl_teacher_main
        + args.lambda_cmnp * loss_cmnp
    )

    loss_staf = soft_dice_loss(
        student_out["staf_logits"],
        teacher_staf_logits.detach(),
        input_from_logits=True,
        target_from_logits=True,
    ) + weighted_ce_dice_loss(
        student_out["staf_logits"],
        pseudo_label,
        pseudo_weight,
        ignore_index=args.ignore_index,
    )
    loss_pseudo = weighted_ce_dice_loss(
        student_out["logits"],
        pseudo_label,
        pseudo_weight,
        ignore_index=args.ignore_index,
    )
    loss_cons = consistency_loss(
        student_out["logits"],
        teacher_logits.detach(),
        weight_map=pseudo_weight,
        loss_type="mse",
    )

    loss_total = (
        loss_scr
        + args.lambda_epl * loss_epl
        + args.lambda_staf * loss_staf
        + args.lambda_pseudo * loss_pseudo
        + args.lambda_cons * loss_cons
    )

    stats = pseudo_label_stats(
        pseudo_label,
        pseudo_weight,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    metrics = {
        "loss_total": loss_total.detach(),
        "loss_scr": loss_scr.detach(),
        "loss_epl": loss_epl.detach(),
        "loss_epl_student_coarse": loss_epl_student_coarse.detach(),
        "loss_epl_teacher_coarse": loss_epl_teacher_coarse.detach(),
        "loss_epl_teacher_main": loss_epl_teacher_main.detach(),
        "loss_cmnp": loss_cmnp.detach(),
        "loss_staf": loss_staf.detach(),
        "loss_pseudo": loss_pseudo.detach(),
        "loss_cons": loss_cons.detach(),
        "pseudo_valid_ratio": stats["valid_ratio"],
        "pseudo_mean_weight": stats["mean_weight"],
        "cmnp_class_quality": cmnp_info["class_quality"].mean().detach(),
        "pef_reliability": pef_info["pixel_reliability"].mean().detach(),
    }
    class_quality = cmnp_info["class_quality"].detach()
    for class_idx in range(args.num_classes):
        metrics[f"cmnp_class_quality_{class_idx}"] = class_quality[:, class_idx].mean()
    if "class_counts" in stats:
        total_pixels = torch.tensor(
            float(pseudo_label.numel()),
            device=pseudo_label.device,
            dtype=stats["class_counts"].dtype,
        ).clamp_min(1.0)
        for class_idx in range(args.num_classes):
            metrics[f"pseudo_class_count_{class_idx}"] = stats["class_counts"][class_idx]
            metrics[f"pseudo_class_ratio_{class_idx}"] = stats["class_counts"][class_idx] / total_pixels
    metrics.update(pair_metrics)
    if return_visuals:
        metrics["visuals"] = {
            "images": images.detach(),
            "scribble": scribble.detach(),
            "pred": torch.argmax(student_out["logits"].detach(), dim=1),
            "pseudo_label": pseudo_label.detach(),
            "pseudo_weight": pseudo_weight.detach(),
            "edge_map": edge_map.detach(),
            "p_evo": torch.argmax(p_evo.detach(), dim=1),
        }
    return loss_total, metrics


def _normalize_image_for_tb(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    reduce_dims = tuple(range(1, image.ndim))
    min_val = image.amin(dim=reduce_dims, keepdim=True)
    max_val = image.amax(dim=reduce_dims, keepdim=True)
    return (image - min_val) / (max_val - min_val + 1e-6)


def _label_to_rgb(label: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    if label.ndim == 2:
        label = label.unsqueeze(0)
    palette = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.90, 0.10, 0.10],
            [0.10, 0.75, 0.20],
            [0.10, 0.35, 0.90],
            [1.00, 0.90, 0.10],
            [0.75, 0.20, 0.85],
            [0.10, 0.85, 0.85],
            [0.95, 0.45, 0.10],
        ],
        dtype=torch.float32,
        device=label.device,
    )
    max_index = min(max(num_classes, ignore_index + 1), palette.shape[0]) - 1
    safe_label = label.long().clamp(0, max_index)
    rgb = palette[safe_label].permute(0, 3, 1, 2)
    if ignore_index < palette.shape[0]:
        ignore = label == int(ignore_index)
        rgb = torch.where(ignore.unsqueeze(1), torch.ones_like(rgb), rgb)
    return rgb


def _overlay_label(image: torch.Tensor, label_rgb: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    image_rgb = image.repeat(1, 3, 1, 1)
    valid = (label_rgb < 0.98).any(dim=1, keepdim=True)
    overlay = (1.0 - alpha) * image_rgb + alpha * label_rgb
    return torch.where(valid, overlay, image_rgb).clamp(0.0, 1.0)


def log_training_visuals(
    writer: SummaryWriter,
    visuals: Dict[str, torch.Tensor],
    iter_num: int,
    num_classes: int,
    ignore_index: int,
    max_samples: int,
) -> None:
    images = _normalize_image_for_tb(visuals["images"][:max_samples].detach().cpu())
    scribble = visuals["scribble"][:max_samples].detach().cpu()
    pred = visuals["pred"][:max_samples].detach().cpu()
    pseudo_label = visuals["pseudo_label"][:max_samples].detach().cpu()
    pseudo_weight = visuals["pseudo_weight"][:max_samples].detach().cpu().unsqueeze(1)
    edge_map = visuals["edge_map"][:max_samples].detach().cpu()
    p_evo = visuals["p_evo"][:max_samples].detach().cpu()

    scribble_rgb = _label_to_rgb(scribble, num_classes=num_classes, ignore_index=ignore_index)
    pred_rgb = _label_to_rgb(pred, num_classes=num_classes, ignore_index=ignore_index)
    pseudo_rgb = _label_to_rgb(pseudo_label, num_classes=num_classes, ignore_index=ignore_index)
    evo_rgb = _label_to_rgb(p_evo, num_classes=num_classes, ignore_index=ignore_index)

    writer.add_images("vis/image", images, iter_num)
    writer.add_images("vis/scribble_overlay", _overlay_label(images, scribble_rgb), iter_num)
    writer.add_images("vis/pred_overlay", _overlay_label(images, pred_rgb), iter_num)
    writer.add_images("vis/evo_overlay", _overlay_label(images, evo_rgb), iter_num)
    writer.add_images("vis/pseudo_overlay", _overlay_label(images, pseudo_rgb), iter_num)
    writer.add_images("vis/pseudo_weight", pseudo_weight.clamp(0.0, 1.0), iter_num)
    writer.add_images("vis/edge_map", edge_map.clamp(0.0, 1.0), iter_num)


def validate(
    student,
    val_loader: DataLoader,
    args: argparse.Namespace,
    writer: SummaryWriter,
    iter_num: int,
) -> Tuple[float, float]:
    from val_2D import test_single_volume

    student.eval()
    metric_list = 0.0
    for sampled_batch in val_loader:
        metric_i = test_single_volume(
            sampled_batch["image"],
            sampled_batch["label"],
            student,
            classes=args.num_classes,
            patch_size=args.patch_size,
        )
        metric_list += np.array(metric_i)
    metric_list = metric_list / max(1, len(val_loader))
    for class_i in range(args.num_classes - 1):
        writer.add_scalar(f"val/class_{class_i + 1}_dice", metric_list[class_i, 0], iter_num)
        writer.add_scalar(f"val/class_{class_i + 1}_hd95", metric_list[class_i, 1], iter_num)
    performance = float(np.mean(metric_list, axis=0)[0])
    mean_hd95 = float(np.mean(metric_list, axis=0)[1])
    writer.add_scalar("val/mean_dice", performance, iter_num)
    writer.add_scalar("val/mean_hd95", mean_hd95, iter_num)
    logging.info("iteration %d : mean_dice %.6f mean_hd95 %.6f", iter_num, performance, mean_hd95)
    student.train()
    return performance, mean_hd95


def train(args: argparse.Namespace, snapshot_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, teacher, mnp_extractor, cmnp_loss_fn, pef_fusion = build_models_and_modules(args, device)
    optimizer = optim.AdamW(student.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    labeled_loader, unlabeled_loader, val_loader = build_loaders(args)
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))
    max_iterations = args.max_epochs * len(labeled_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=0.0)
    runtime_config = {
        "device": str(device),
        "labeled_slices": len(labeled_loader.dataset),
        "unlabeled_slices": len(unlabeled_loader.dataset) if unlabeled_loader is not None else 0,
        "validation_volumes": len(val_loader.dataset),
        "max_iterations": max_iterations,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    save_run_config(snapshot_path, args, extra=runtime_config)

    student.train()
    best_performance = 0.0
    best_mean_hd95 = float("inf")
    iter_num = 0
    iterator = tqdm(range(args.max_epochs), ncols=90)
    unlabeled_iter = cycle(unlabeled_loader) if unlabeled_loader is not None else None

    for _ in iterator:
        for labeled_batch in labeled_loader:
            unlabeled_batch = next(unlabeled_iter) if unlabeled_iter is not None else None
            images, scribble = prepare_batch(labeled_batch, unlabeled_batch, args, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                need_visuals = args.vis_interval > 0 and (iter_num + 1) % args.vis_interval == 0
                loss, metrics = compute_train_step(
                    images,
                    scribble,
                    student,
                    teacher,
                    mnp_extractor,
                    cmnp_loss_fn,
                    pef_fusion,
                    args,
                    return_visuals=need_visuals,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema_model(student, teacher, decay=args.ema_decay, global_step=iter_num)

            iter_num += 1
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", lr, iter_num)
            for key, value in metrics.items():
                if key == "visuals":
                    continue
                writer.add_scalar(f"train/{key}", float(value), iter_num)

            if "visuals" in metrics:
                log_training_visuals(
                    writer,
                    metrics["visuals"],
                    iter_num,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    max_samples=args.vis_num,
                )

            if iter_num % 20 == 0:
                logging.info(
                    "iteration %d : loss %.6f scr %.6f epl %.6f cmnp %.6f pseudo %.6f",
                    iter_num,
                    float(metrics["loss_total"]),
                    float(metrics["loss_scr"]),
                    float(metrics["loss_epl"]),
                    float(metrics["loss_cmnp"]),
                    float(metrics["loss_pseudo"]),
                )

            if iter_num > 0 and iter_num % args.val_interval == 0:
                performance, mean_hd95 = validate(student, val_loader, args, writer, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_path = os.path.join(snapshot_path, f"iter_{iter_num}_dice_{best_performance:.4f}.pth")
                    best_path = os.path.join(snapshot_path, "erecu_med_best_model.pth")
                    torch.save(
                        {
                            "student_state_dict": student.state_dict(),
                            "teacher_state_dict": teacher.state_dict(),
                            "iter_num": iter_num,
                            "best_performance": best_performance,
                            "mean_hd95": mean_hd95,
                            "args": vars(args),
                        },
                        save_path,
                    )
                    torch.save(student.state_dict(), best_path)
                    save_checkpoint_metadata(
                        best_path,
                        args,
                        iter_num=iter_num,
                        performance=performance,
                        mean_hd95=mean_hd95,
                        selection_metric="best_val_mean_dice",
                    )
                    logging.info("save best dice checkpoint to %s", save_path)

                if mean_hd95 < best_mean_hd95:
                    best_mean_hd95 = mean_hd95
                    save_path = os.path.join(snapshot_path, f"iter_{iter_num}_hd95_{best_mean_hd95:.4f}.pth")
                    best_path = os.path.join(snapshot_path, "best_mean_hd95.pth")
                    torch.save(
                        {
                            "student_state_dict": student.state_dict(),
                            "teacher_state_dict": teacher.state_dict(),
                            "iter_num": iter_num,
                            "performance": performance,
                            "best_mean_hd95": best_mean_hd95,
                            "args": vars(args),
                        },
                        save_path,
                    )
                    torch.save(student.state_dict(), best_path)
                    save_checkpoint_metadata(
                        best_path,
                        args,
                        iter_num=iter_num,
                        performance=performance,
                        mean_hd95=mean_hd95,
                        selection_metric="best_val_mean_hd95",
                    )
                    logging.info("save best HD95 checkpoint to %s", save_path)

            if iter_num > 0 and iter_num % args.save_interval == 0:
                save_path = os.path.join(snapshot_path, f"iter_{iter_num}.pth")
                torch.save(
                    {
                        "student_state_dict": student.state_dict(),
                        "teacher_state_dict": teacher.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "args": vars(args),
                    },
                    save_path,
                )
                logging.info("save checkpoint to %s", save_path)

    runtime_config.update(
        {
            "completed_iterations": iter_num,
            "best_val_mean_dice": best_performance,
            "best_val_mean_hd95": best_mean_hd95,
        }
    )
    save_run_config(snapshot_path, args, extra=runtime_config)
    writer.close()
    logging.info(
        "training finished, best mean dice %.6f best mean hd95 %.6f",
        best_performance,
        best_mean_hd95,
    )


def dry_run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.patch_size = [args.dry_run_size, args.dry_run_size]
    args.batch_size = 2
    args.labeled_bs = 2
    student, teacher, mnp_extractor, cmnp_loss_fn, pef_fusion = build_models_and_modules(args, device)
    optimizer = optim.AdamW(student.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    images = torch.randn(args.batch_size, 1, args.dry_run_size, args.dry_run_size, device=device)
    scribble = torch.full(
        (args.batch_size, args.dry_run_size, args.dry_run_size),
        int(args.ignore_index),
        dtype=torch.long,
        device=device,
    )
    scribble[:, 4:8, 4:8] = 1
    scribble[:, 12:16, 12:16] = 2

    loss, metrics = compute_train_step(
        images,
        scribble,
        student,
        teacher,
        mnp_extractor,
        cmnp_loss_fn,
        pef_fusion,
        args,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    update_ema_model(student, teacher, decay=args.ema_decay, global_step=0)
    print("dry_run ok")
    for key, value in metrics.items():
        print(f"{key}: {float(value):.6f}")


def main() -> None:
    args = parse_args()
    args.deep_to_shallow_pairs_resolved = parse_deep_to_shallow_pairs(args.deep_to_shallow_pairs)
    args.mnp_channels, native_features = resolve_mnp_metadata(args.mnp_variant)
    args.native_features = list(native_features)
    setup_seed(args)
    if args.dry_run:
        dry_run(args)
        return

    snapshot_path = make_snapshot_path(args)
    configure_logging(snapshot_path, args)
    copy_code_snapshot(snapshot_path)
    save_run_config(snapshot_path, args)
    train(args, snapshot_path)


if __name__ == "__main__":
    main()
