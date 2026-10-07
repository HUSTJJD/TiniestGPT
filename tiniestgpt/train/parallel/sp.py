"""序列并行（SP）：沿 **序列维** 切分，专门用来削掉 TP 区之外的激活显存。

Megatron-LM 的观察：TP 只对 **attention 和 FFN 的矩阵乘** 有收益，
而 LayerNorm / Dropout / 残差加这些"逐元素算子"在 TP 下是**每张卡各算一份**，
它们的激活显存完全没有省下来。于是把这些区域改成沿序列维切分：

```
  输入 [B, T, C]  (完整)
      │  reduce-scatter / chunk     ← 进入 SP 区
   [B, T/N, C]   LayerNorm / Dropout / 残差
      │  all-gather                ← 进入 TP 区（attention / FFN 需要完整序列）
   [B, T, C]     attention / FFN（内部仍按 head / hidden 切）
      │  reduce-scatter            ← 回到 SP 区
   [B, T/N, C]
```

通信量：all-gather + reduce-scatter 各一次，合计 2·B·T·C，
与 TP 区内那 2 次 all-reduce 相当——所以 SP 是"用同样的通信量换激活显存"。
"""

from __future__ import annotations

from typing import Optional

import torch

from .comm import DistContext

__all__ = ["ScatterToSequenceParallel", "GatherFromSequenceParallel",
           "scatter_to_sp", "gather_from_sp", "reduce_scatter_to_sp",
           "sequence_parallel_memory_factor"]


class _ScatterToSP(torch.autograd.Function):
    """forward: 完整 → 分片；backward: 分片 → 完整（all-gather）。"""

    @staticmethod
    def forward(ctx_, x: torch.Tensor, dim: int, world_size: int, rank: int) -> torch.Tensor:
        ctx_.dim, ctx_.world_size, ctx_.rank = dim, world_size, rank
        return x.chunk(world_size, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx_, grad: torch.Tensor):
        # 梯度需要拼回完整形状
        return torch.cat([grad] * ctx_.world_size, dim=ctx_.dim), None, None, None


class _GatherFromSP(torch.autograd.Function):
    """forward: 分片 → 完整（all-gather）；backward: 完整 → 分片（chunk）。"""

    @staticmethod
    def forward(ctx_, x: torch.Tensor, dim: int, world_size: int, rank: int) -> torch.Tensor:
        ctx_.dim, ctx_.world_size, ctx_.rank = dim, world_size, rank
        return torch.cat([x] * world_size, dim=dim)

    @staticmethod
    def backward(ctx_, grad: torch.Tensor):
        return grad.chunk(ctx_.world_size, dim=ctx_.dim)[ctx_.rank].contiguous(), None, None, None


ScatterToSequenceParallel = _ScatterToSP.apply
GatherFromSequenceParallel = _GatherFromSP.apply


def scatter_to_sp(x: torch.Tensor, ctx: DistContext, dim: int = 1) -> torch.Tensor:
    """完整张量 → 本 rank 的序列分片。"""
    if not ctx.parallel:
        return x
    if ctx.simulated:
        return _ScatterToSP.apply(x, dim, ctx.world_size, 0)   # 模拟模式固定取第 0 片
    return _ScatterToSP.apply(x, dim, ctx.world_size, ctx.rank)


def gather_from_sp(x: torch.Tensor, ctx: DistContext, dim: int = 1) -> torch.Tensor:
    """序列分片 → 完整张量（真实分布式下是一次 all-gather）。"""
    if not ctx.parallel:
        return x
    if not ctx.simulated:
        return ctx.all_gather(x.contiguous(), dim=dim)
    return _GatherFromSP.apply(x, dim, ctx.world_size, 0)


def reduce_scatter_to_sp(x: torch.Tensor, ctx: DistContext, dim: int = 1) -> torch.Tensor:
    """TP 区结束后的求和 + 切分（真实分布式下是一次 reduce-scatter）。"""
    if not ctx.parallel:
        return x
    if not ctx.simulated:
        return ctx.reduce_scatter(x.contiguous(), dim=dim)
    return _ScatterToSP.apply(x, dim, ctx.world_size, 0)


def sequence_parallel_memory_factor(num_layers: int) -> float:
    """粗略估计：SP 把"非 TP 区"的激活显存降到 1/N（N=TP degree）。

    这里返回"可被 SP 削减的激活占比"的经验值：每个 block 里
    LayerNorm(2) + Dropout(1) + 残差(2) 大约占激活的一半左右。
    """
    return 0.5 if num_layers > 0 else 0.0
