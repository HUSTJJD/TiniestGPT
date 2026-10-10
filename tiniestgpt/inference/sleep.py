"""Sleep mode：让同一张 GPU 在**训练**与**推理**之间来回切换。

RL 后训练（GRPO / PPO）的循环是：

    rollout（推理引擎采样） → 打分 → 训练（Megatron/FSDP 更新权重） → 同步权重 → 再 rollout

如果在训练和推理之间不做处理，两边都要常驻显存，显存直接翻倍。
vLLM 的 sleep mode 解决这个问题，分两个级别：

* **level 1**：卸载**模型权重**到 CPU，丢弃 KV Cache（显存回收 ~70%）；
* **level 2**：连 KV Cache 的**缓冲区**也释放（回收 ~90%+），
  唤醒时需要重新分配并重建 CUDA Graph。

代价：唤醒要重新搬运权重（几十 GB 的 PCIe 传输，秒级），
所以只有"训练一轮要几分钟、rollout 只占几秒"时才划算。

本模块给出与引擎解耦的最小实现：快照 → 卸载 → 唤醒 → 恢复。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["SleepLevel", "SleepManager", "weight_bytes"]


class SleepLevel:
    AWAKE = 0
    UNLOAD_WEIGHTS = 1        # level 1
    RELEASE_BUFFERS = 2       # level 2


def weight_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


@dataclass
class SleepManager:
    """管理一个引擎/模型的睡眠与唤醒。"""

    model: nn.Module
    level: int = SleepLevel.AWAKE
    cpu_snapshot: Dict[str, torch.Tensor] = field(default_factory=dict)
    dropped_buffers: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    def sleep(self, level: int = SleepLevel.UNLOAD_WEIGHTS) -> float:
        """进入睡眠，返回耗时（秒）。"""
        t0 = time.time()
        if level >= SleepLevel.UNLOAD_WEIGHTS and not self.cpu_snapshot:
            for n, p in self.model.named_parameters():
                self.cpu_snapshot[n] = p.detach().to("cpu", copy=True)
            for n, p in self.model.named_parameters():
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        if level >= SleepLevel.RELEASE_BUFFERS:
            for n, b in list(self.model.named_buffers()):
                if b.numel() > 0:
                    self.dropped_buffers.append(n)
            if hasattr(self.model, "reset"):
                try:
                    self.model.reset()            # 释放 KV Cache / block allocator
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.level = level
        self.stats["sleep_s"] = time.time() - t0
        return self.stats["sleep_s"]

    def wake_up(self) -> float:
        """唤醒，返回耗时（秒）——这一步要付 PCIe 传输成本。"""
        t0 = time.time()
        if self.cpu_snapshot:
            for n, p in self.model.named_parameters():
                if n in self.cpu_snapshot:
                    p.data = self.cpu_snapshot[n].to(p.device)
            self.cpu_snapshot.clear()
        self.dropped_buffers.clear()
        self.level = SleepLevel.AWAKE
        self.stats["wake_s"] = time.time() - t0
        return self.stats["wake_s"]

    def sync_weights_from(self, src: nn.Module) -> int:
        """训练结束后把权重同步回推理引擎（省掉一次 CPU 往返时可用）。"""
        n = 0
        with torch.no_grad():
            for p_dst, p_src in zip(self.model.parameters(), src.parameters()):
                if p_dst.shape == p_src.shape:
                    p_dst.data.copy_(p_src.data.to(p_dst.device))
                    n += 1
        return n

    def report(self) -> str:
        return (f"SleepManager(level={self.level}): 权重 {weight_bytes(self.model) / 1e6:.0f} MB，"
                f"睡眠 {self.stats.get('sleep_s', 0):.2f}s / 唤醒 {self.stats.get('wake_s', 0):.2f}s")
