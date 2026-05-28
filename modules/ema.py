"""EMA teacher utilities for EReCu-Med."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


def create_ema_model(student: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    """Create an EMA teacher initialized from the student."""
    teacher = copy.deepcopy(student)
    if device is not None:
        teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.detach_()
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_model(
    student: nn.Module,
    teacher: nn.Module,
    decay: float = 0.99,
    global_step: Optional[int] = None,
    update_buffers: bool = True,
) -> None:
    """Update teacher parameters with exponential moving average."""
    decay = float(decay)
    if global_step is not None:
        decay = min(decay, 1.0 - 1.0 / float(global_step + 1))

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name]
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value.detach(), alpha=1.0 - decay)
        elif update_buffers:
            teacher_value.copy_(student_value.detach())


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(bool(requires_grad))


def set_ema_eval(teacher: nn.Module) -> nn.Module:
    teacher.eval()
    set_requires_grad(teacher, False)
    return teacher


__all__ = [
    "create_ema_model",
    "set_ema_eval",
    "set_requires_grad",
    "update_ema_model",
]
