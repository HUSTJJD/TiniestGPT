"""Checkpoint：保存 / 恢复 / 异步落盘。

工程要点：
  * 只保存 **rank0**（或 FSDP 的 full state dict），避免 N 份重复写盘；
  * **异步保存**：训练不必等磁盘 —— 用后台线程 + ``tensor.clone()`` 快照，
    把"几百毫秒到几秒的写盘"从关键路径上移除；
  * 保存 optimizer 状态时注意 Muon 的动量缓冲（体积≈参数量），
    若磁盘紧张可用 ``save_optimizer=False``。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..common.logging import get_logger

log = get_logger("tiniestgpt.ckpt")

__all__ = ["save_checkpoint", "load_checkpoint", "find_latest"]


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    step: int = 0,
    cfg: Any = None,
    extra: Optional[Dict] = None,
    async_save: bool = False,
    save_optimizer: bool = True,
) -> Path:
    """保存训练状态到 ``path``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = _unwrap(model)

    payload: Dict[str, Any] = {
        "step": step,
        "model": {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()},
        "config": cfg if isinstance(cfg, dict) else (vars(cfg) if cfg is not None else None),
        "extra": extra or {},
    }
    if save_optimizer and optimizer is not None:
        payload["optimizer"] = {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                                for k, v in optimizer.state_dict().items()}
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()

    if async_save:
        def _worker():
            t0 = time.time()
            torch.save(payload, path)
            log.info("async checkpoint saved -> %s (%.1fs)", path, time.time() - t0)

        threading.Thread(target=_worker, daemon=True).start()
        return path

    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    m = _unwrap(model)
    sd = ckpt["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        log.warning("missing keys: %d (%s...)", len(missing), missing[:5])
    if unexpected:
        log.warning("unexpected keys: %d (%s...)", len(unexpected), unexpected[:5])
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            log.warning("恢复优化器状态失败: %s", exc)
    if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            log.warning("恢复调度器状态失败: %s", exc)
    return ckpt


def find_latest(dir_path: str | Path, suffix: str = ".pt") -> Optional[Path]:
    """在目录里找 step 最大的 checkpoint（命名形如 ``ckpt_0001000.pt``）。"""
    d = Path(dir_path)
    if not d.exists():
        return None
    cands = sorted(d.glob(f"*{suffix}"))
    return cands[-1] if cands else None
