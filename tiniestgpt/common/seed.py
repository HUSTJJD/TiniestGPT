"""随机性与分布式环境工具。"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed", "get_rank", "get_local_rank", "get_world_size", "is_main_process", "init_device"]


def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))


def is_main_process() -> bool:
    return get_rank() == 0


def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定所有随机源。``deterministic=True`` 会牺牲速度换取可复现（CuDNN 确定性算法）。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # pragma: no cover
        pass


def init_device(prefer_gpu: bool = True) -> "object":
    """返回一个可用的 torch.device；无 CUDA 时自动退回 CPU。"""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        local_rank = get_local_rank()
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")
