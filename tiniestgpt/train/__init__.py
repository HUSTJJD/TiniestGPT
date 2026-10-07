"""预训练工程层。

覆盖：优化器（AdamW / Muon / Sophia-G / Lion）、学习率调度（WSD / cosine / trapezoid）、
损失函数（CE + label smoothing + z-loss + MoE 辅助损失）、
混合精度、梯度裁剪、DDP/FSDP、checkpoint、吞吐与 MFU 统计。
"""

from .config import TrainConfig
from .optim import build_optimizer, Muon, SophiaG, Lion
from .lr_sched import build_scheduler
from .losses import language_modeling_loss
from .engine import Trainer

__all__ = [
    "TrainConfig", "build_optimizer", "Muon", "SophiaG", "Lion",
    "build_scheduler", "language_modeling_loss", "Trainer",
]
