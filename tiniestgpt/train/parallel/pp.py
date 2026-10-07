"""流水线并行（PP）：把**层**切到不同设备上，用微批次填满"气泡"。

为什么需要 PP：
模型层数 × 参数量太大，一张卡放不下；TP 又只能机内切。
PP 把 layer 0~k 放设备 A、k+1~2k 放设备 B……像工厂流水线一样传递激活。

核心矛盾是 **bubble（气泡）**：
流水线的开头（warmup）和结尾（cooldown）总有设备空转。
微批次（micro-batch）越多，气泡占比越低：

```
bubble_ratio ≈ (num_stages - 1) / num_microbatches
```

**1F1B**（One-Forward-One-Backward）是标准调度：
先 warmup 灌入 ``num_stages - rank - 1`` 个微批次，
之后每个 step 严格"做一个 forward、做一个 backward"，
把激活显存从 O(num_microbatches) 压到 O(num_stages)——这是它能跑大模型的关键。

本文件在**单进程**里把各 stage 的 forward 串行跑一遍（真实分布式下
stage 之间靠 send/recv 传激活），因此数值上等价于顺序执行，
但调度顺序、气泡分析、微批次切分都是真的。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn

from .comm import DistContext

__all__ = ["PipelineParallel", "bubble_ratio", "make_stages", "PipelineStats"]


def bubble_ratio(num_stages: int, num_microbatches: int) -> float:
    """1F1B 的气泡占比（理想模型）。"""
    if num_stages <= 1:
        return 0.0
    return (num_stages - 1) / max(num_microbatches, 1)


def make_stages(layers: Sequence[nn.Module], num_stages: int) -> List[nn.ModuleList]:
    """把 ``layers`` 尽量均匀地切成 ``num_stages`` 段。"""
    num_stages = max(1, min(num_stages, len(layers)))
    per = len(layers) / num_stages
    out: List[nn.ModuleList] = []
    for s in range(num_stages):
        lo, hi = int(round(s * per)), int(round((s + 1) * per))
        out.append(nn.ModuleList(list(layers[lo:hi])))
    return out


class PipelineStats(dict):
    """调度统计：便于在日志里看气泡与微批次情况。"""


class PipelineParallel(nn.Module):
    """把一串层按 stage 切开，用 1F1B 调度跑微批次。

    :param layers:        连续的层（如 ``model.layers``）
    :param num_stages:    stage 数（真实场景 = PP degree）
    :param num_microbatches: 一个 mini-batch 切成多少微批次
    """

    def __init__(self, layers: Sequence[nn.Module], num_stages: int = 2,
                 num_microbatches: int = 4, ctx: Optional[DistContext] = None,
                 stage_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__()
        self.num_stages = max(1, int(num_stages))
        self.num_microbatches = max(1, int(num_microbatches))
        self.ctx = ctx or DistContext(world_size=1)
        self.stages = nn.ModuleList(make_stages(layers, self.num_stages))
        self._stage_fn = stage_fn
        self.stats = PipelineStats(
            num_stages=self.num_stages,
            num_microbatches=self.num_microbatches,
            bubble_ratio=bubble_ratio(self.num_stages, self.num_microbatches),
        )

    # ------------------------------------------------------------------ #
    def _run_stage(self, s: int, x: torch.Tensor) -> torch.Tensor:
        if self._stage_fn is not None:
            return self._stage_fn(s, x)
        for layer in self.stages[s]:
            out = layer(x)
            # TransformerBlock 返回 (tensor, aux_loss)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: [num_microbatches * b, T, C] → 同形状输出。

        按**时间步**推进流水线：时间步 ``t`` 时，stage ``s`` 处理的是
        微批次 ``t - s``。这样同一时刻最多有 ``num_stages`` 个微批次在流水线上，
        与真实 1F1B 的前向阶段一致（真实系统里各 stage 在不同设备上并行）。
        """
        mb = self.num_microbatches
        if x.shape[0] % mb != 0:
            raise ValueError(f"batch={x.shape[0]} 不能被 num_microbatches={mb} 整除")

        chunks = list(x.chunk(mb, dim=0))
        results: List[Optional[torch.Tensor]] = [None] * mb
        carry: dict[tuple[int, int], torch.Tensor] = {}

        for t in range(mb + self.num_stages - 1):
            for s in range(self.num_stages):
                i = t - s
                if not (0 <= i < mb):
                    continue
                h = chunks[i] if s == 0 else carry.pop((i, s - 1))
                h = self._run_stage(s, h)
                if s == self.num_stages - 1:
                    results[i] = h
                else:
                    carry[(i, s)] = h
        return torch.cat(results, dim=0)
