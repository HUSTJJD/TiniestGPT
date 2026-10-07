"""通信抽象：**真实分布式** 与 **单机模拟多卡** 两套实现。

为什么要有"模拟"模式：
3D 并行、ZeRO 这些概念的核心是"张量怎么切、通信量是多少"，
但大多数学习者手上只有一张卡。所以这里提供一个 ``DistContext``：

* ``world_size=1``        → 所有集合通信都是恒等变换，并行模块退化为普通模块；
* 有 ``torch.distributed`` → 走真正的 NCCL/gloo；
* ``world_size>1`` 但没起进程组 → **模拟模式**：依次扮演每个 rank 跑一遍
  （见 :func:`run_ranks`），集合通信用累加/拼接在单进程内算出真实结果。

模拟模式下只有**最后一个 rank** 的返回值是完全正确的（前面的 rank 只保证
形状正确），所以 :func:`run_ranks` 只返回最后一次调用的结果。
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

__all__ = ["DistContext", "run_ranks", "is_distributed_ready"]


def is_distributed_ready() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


class DistContext:
    """一次并行计算需要的全部上下文：world_size / rank / 集合通信实现。"""

    def __init__(self, world_size: int = 1, rank: int = 0, process_group=None) -> None:
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.process_group = process_group
        # 模拟器状态
        self._acc: Optional[torch.Tensor] = None
        self._acc_count = 0
        self._gathered: List[torch.Tensor] = []

    @property
    def simulated(self) -> bool:
        """world_size>1 但没起进程组 → 用单进程模拟。"""
        return self.world_size > 1 and not is_distributed_ready()

    @property
    def parallel(self) -> bool:
        return self.world_size > 1

    # ------------------------------------------------------------------ #
    def all_reduce(self, t: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """所有 rank 的张量求和（DDP 里再除以 world_size 就是平均）。"""
        if not self.parallel:
            return t
        if not self.simulated:
            import torch.distributed as dist

            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.process_group)
            return t
        self._acc = t.detach().clone() if self._acc is None else self._acc + t.detach()
        self._acc_count += 1
        out = self._acc
        if self._acc_count >= self.world_size:
            self._acc, self._acc_count = None, 0
        return out

    def all_gather(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """把各 rank 的分片拼回完整张量（每个 rank 拿到的结果相同）。"""
        if not self.parallel:
            return t
        if not self.simulated:
            import torch.distributed as dist

            parts = [torch.empty_like(t) for _ in range(self.world_size)]
            dist.all_gather(parts, t.contiguous(), group=self.process_group)
            return torch.cat(parts, dim=dim)
        self._gathered.append(t.detach().clone())
        if len(self._gathered) >= self.world_size:
            out = torch.cat(self._gathered, dim=dim)
            self._gathered.clear()
            return out
        return t

    def reduce_scatter(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """先 all_reduce 再按 dim 切分，本 rank 只留自己那一片。"""
        if not self.parallel:
            return t
        shard = lambda x: x.chunk(self.world_size, dim=dim)[self.rank]  # noqa: E731
        if not self.simulated:
            import torch.distributed as dist

            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.process_group)
            return shard(t).contiguous()
        self._acc = t.detach().clone() if self._acc is None else self._acc + t.detach()
        self._acc_count += 1
        out = shard(self._acc)
        if self._acc_count >= self.world_size:
            self._acc, self._acc_count = None, 0
        return out

    def broadcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not self.parallel:
            return t
        if not self.simulated:
            import torch.distributed as dist

            dist.broadcast(t, src=src, group=self.process_group)
        return t

    def reset(self) -> None:
        self._acc, self._acc_count, self._gathered = None, 0, []


def run_ranks(ctx: DistContext, fn: Callable[[int], object],
              ranks: Optional[List[int]] = None) -> object:
    """依次扮演每个 rank 执行 ``fn(rank)``，返回**最后一个** rank 的结果。

    这是模拟模式的关键：单进程把 N 张卡的计算串行跑一遍，
    集合通信由 :class:`DistContext` 在内部累加/拼接，
    于是最后一个 rank 拿到的就是"多卡并行"的真实结果。
    """
    out = None
    original = ctx.rank
    for r in (ranks or list(range(ctx.world_size))):
        ctx.rank = r
        out = fn(r)
    ctx.rank = original
    return out
