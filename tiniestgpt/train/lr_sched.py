"""学习率调度。

**WSD（Warmup-Stable-Decay）** 是近年大模型预训练的事实标准（MiniCPM、DeepSeek、Llama3 都用）：
  * warmup：线性升到峰值；
  * stable：**保持常数**——这让你可以随时"接着训"，
    不必事先承诺训练多少步（对比 cosine 必须提前定死总步数）；
  * decay：在**任意时刻**启动一段短衰减即可得到最优模型。
    WSD 的一个漂亮性质是：**衰减中途的 checkpoint 也能直接用**，
    因此可以做"中途退火 + 早停"的实验循环（loss 曲线可直接对比）。

decay 的形状可选：linear / cosine / sqrt / 1-sqrt（1-sqrt 前期掉得快，常见且好用）。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

__all__ = ["LRScheduler", "build_scheduler"]


class LRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, schedule: str = "wsd",
                 warmup_steps: int = 100, total_steps: int = 1000,
                 min_lr_ratio: float = 0.05, decay_fraction: float = 0.2,
                 decay_type: str = "1-sqrt", decay_start: Optional[int] = None) -> None:
        self.optimizer = optimizer
        self.schedule = schedule.lower()
        self.warmup = max(int(warmup_steps), 0)
        self.total = max(int(total_steps), 1)
        self.min_ratio = min_lr_ratio
        self.decay_fraction = decay_fraction
        self.decay_type = decay_type
        self.step_count = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._decay_start = decay_start if decay_start is not None else self._default_decay_start()

    def _default_decay_start(self) -> int:
        if self.schedule == "wsd":
            return int(self.total * (1.0 - self.decay_fraction))
        return self.total

    # ------------------------------------------------------------------ #
    def start_decay(self, at_step: Optional[int] = None) -> None:
        """WSD 的灵魂：随时"开始退火"。"""
        self._decay_start = (self.step_count if at_step is None else at_step) + 1
        self.total = self._decay_start + max(int(self.total * self.decay_fraction), 1)

    @property
    def decay_start(self) -> int:
        return self._decay_start

    # ------------------------------------------------------------------ #
    def _mult(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / max(self.warmup, 1)
        if self.schedule == "constant":
            return 1.0
        if self.schedule == "linear":
            t = (step - self.warmup) / max(self.total - self.warmup, 1)
            return self.min_ratio + (1 - self.min_ratio) * max(1.0 - t, 0.0)
        if self.schedule == "cosine":
            t = (step - self.warmup) / max(self.total - self.warmup, 1)
            t = min(max(t, 0.0), 1.0)
            return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        if self.schedule == "trapezoid":
            # 线性升 → 平台 → 线性降
            if step < self.decay_start:
                return 1.0
            t = (step - self.decay_start) / max(self.total - self.decay_start, 1)
            return self.min_ratio + (1 - self.min_ratio) * max(1.0 - min(t, 1.0), 0.0)
        # ---- WSD ----
        if step < self.decay_start:
            return 1.0
        t = (step - self.decay_start) / max(self.total - self.decay_start, 1)
        t = min(max(t, 0.0), 1.0)
        dt = self.decay_type
        if dt == "linear":
            f = 1.0 - t
        elif dt == "cosine":
            f = 0.5 * (1 + math.cos(math.pi * t))
        elif dt == "sqrt":
            f = 1.0 - math.sqrt(t)
        else:                       # "1-sqrt"
            f = math.sqrt(max(1.0 - t, 0.0))
        return self.min_ratio + (1 - self.min_ratio) * f

    # ------------------------------------------------------------------ #
    def get_last_lr(self) -> List[float]:
        m = self._mult(self.step_count)
        return [b * m for b in self.base_lrs]

    def step(self) -> None:
        self.step_count += 1
        m = self._mult(self.step_count)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * m

    def state_dict(self) -> dict:
        return {"step": self.step_count, "decay_start": self._decay_start,
                "total": self.total, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd: dict) -> None:
        self.step_count = sd.get("step", 0)
        self._decay_start = sd.get("decay_start", self._decay_start)
        self.total = sd.get("total", self.total)
        if sd.get("base_lrs"):
            self.base_lrs = sd["base_lrs"]


def build_scheduler(cfg, optimizer: torch.optim.Optimizer) -> LRScheduler:
    return LRScheduler(
        optimizer, schedule=cfg.lr_schedule, warmup_steps=cfg.warmup_steps,
        total_steps=cfg.max_steps, min_lr_ratio=cfg.min_lr_ratio,
        decay_fraction=cfg.decay_fraction, decay_type=cfg.decay_type,
    )
