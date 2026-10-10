"""EMA 权重平均 + loss spike 自动回滚 + 异步 checkpoint。

三件事看起来不相干，其实都是同一个主题的**工程兜底**：

* **EMA**：维护一份指数滑动平均的权重，通常比最后一步的权重更"平"，
  推理/评测时用它可以白拿一点稳定性。代价是一份额外显存。
* **spike 回滚**：大规模训练里 loss spike 是常态而不是意外。
  检测到 spike 就**回退到上一个"干净"的 checkpoint** 并跳过若干步的数据，
  比人工盯着 loss 曲线靠谱。
* **异步 checkpoint**：同步写盘会在每 N 步插一个几秒的停顿；
  把 state_dict 拷到 CPU 后交给后台线程写，训练循环不用等。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["EMA", "SpikeGuard", "AsyncCheckpointer"]


# --------------------------------------------------------------------------- #
class EMA:
    """指数滑动平均权重。

    :param decay: 衰减率。步数少时用 0.99，长训练常用 0.999~0.9999。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 device: Optional[torch.device] = None) -> None:
        self.decay = float(decay)
        self.device = device
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone().to(device) if device else p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))   # 前期加速收敛
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.shadow:
                continue
            self.shadow[n].mul_(d).add_(p.detach().to(self.shadow[n].device), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """把 EMA 权重**临时**装到模型上（评测用）。"""
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters()
                        if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if getattr(self, "_backup", None):
            for n, p in model.named_parameters():
                if n in self._backup:
                    p.data.copy_(self._backup[n])
            self._backup = None

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {n: t.detach().cpu() for n, t in self.shadow.items()}


# --------------------------------------------------------------------------- #
@dataclass
class SpikeGuard:
    """检测 loss spike 并请求回滚。

    判据用**相对阈值**而不是绝对值：
    当前 loss 超过最近窗口均值的 ``factor`` 倍，就判为 spike。
    """

    factor: float = 2.0
    window: int = 50
    cooldown: int = 20          # 回滚后多少步内不再触发
    history: List[float] = field(default_factory=list)
    spikes: int = 0
    _cool: int = 0

    def observe(self, loss: float) -> bool:
        """返回 True 表示**检测到 spike，建议回滚**。"""
        if self._cool > 0:
            self._cool -= 1
            self.history.append(loss)
            return False
        self.history.append(loss)
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < max(self.window // 2, 5):
            return False
        ref = sum(self.history[:-1]) / max(len(self.history) - 1, 1)
        if loss > ref * self.factor and ref > 0:
            self.spikes += 1
            self._cool = self.cooldown
            self.history.clear()
            return True
        return False

    def report(self) -> str:
        return f"SpikeGuard: 触发 {self.spikes} 次 (factor={self.factor}, window={self.window})"


# --------------------------------------------------------------------------- #
class AsyncCheckpointer:
    """后台线程写盘，训练循环不被 IO 阻塞。

    用法::

        ckpt = AsyncCheckpointer()
        ckpt.save(state_dict, "out/last.pt")     # 立即返回
        ...
        ckpt.wait()                              # 退出前务必等它写完
    """

    def __init__(self, max_pending: int = 2) -> None:
        self.max_pending = max_pending
        self._threads: List[threading.Thread] = []
        self.saved = 0
        self.last_error: Optional[str] = None

    def _to_cpu(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=True)
        return obj

    def save(self, state: Dict, path: str) -> bool:
        if len([t for t in self._threads if t.is_alive()]) >= self.max_pending:
            # 上一份还没写完就先不排队，避免内存堆积
            return False
        cpu_state = self._to_cpu(state)
        t = threading.Thread(target=self._worker, args=(cpu_state, path), daemon=True)
        t.start()
        self._threads.append(t)
        self._threads = [x for x in self._threads if x.is_alive()] or [t]
        return True

    def _worker(self, state: Dict, path: str) -> None:
        try:
            torch.save(state, path)
            self.saved += 1
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{path}: {exc}"

    def wait(self, timeout: float = 60.0) -> None:
        end = time.time() + timeout
        for t in list(self._threads):
            t.join(max(end - time.time(), 0.0))

    def report(self) -> str:
        return f"AsyncCheckpoint: 已写 {self.saved} 份" + \
               (f"，最后错误 {self.last_error}" if self.last_error else "")
