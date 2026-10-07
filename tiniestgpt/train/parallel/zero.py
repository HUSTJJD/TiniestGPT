"""ZeRO（Zero Redundancy Optimizer）的**手写实现**：用通信换显存。

显存账本先摆清楚（7B 模型、AdamW、混合精度，单位：字节/参数）：

| 内容 | 精度 | 字节/参数 | 7B 总量 |
|---|---|---|---|
| 参数（计算用） | bf16/fp16 | 2 | 14 GB |
| 梯度 | bf16/fp16 | 2 | 14 GB |
| FP32 主权重 | fp32 | 4 | 28 GB |
| 一阶动量 m | fp32 | 4 | 28 GB |
| 二阶动量 v | fp32 | 4 | 28 GB |
| **合计** | | **16** | **112 GB** |

单卡 80GB 连"参数+梯度+优化器状态"都放不下 —— 注意优化器状态（84GB）
比参数本身（14GB）大 6 倍，所以 **ZeRO 第一刀必须砍在优化器状态上**。

三个阶段切的正是这三类状态：

* **ZeRO-1**：只切**优化器状态**（m / v / fp32 主权重）。
  梯度仍走 all-reduce（和 DDP 一样），只是每张卡只更新自己那 1/N 的参数，
  更新完再 broadcast 给其他卡。
* **ZeRO-2**：再切**梯度**。反向时用 reduce-scatter，每张卡只拿到
  自己负责那 1/N 参数的梯度（而不是完整梯度）。
* **ZeRO-3**：连**参数**也切。forward / backward 时按需 all-gather，用完即弃。
  显存降到 1/N，但通信量约翻倍。

本实现按**整个参数**做 round-robin 分片（DeepSpeed 实际按 flatten 后的 bucket 切，
工程上更省通信，但按参数切更容易看懂）。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .comm import DistContext

__all__ = ["ZeroOptimizer", "partition_params", "zero_memory_report"]


def partition_params(params: Iterable[nn.Parameter], world_size: int,
                     rank: int) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """round-robin 分片，返回 ``(本 rank 拥有的, 其余的)``。"""
    all_params = list(params)
    owned = [p for i, p in enumerate(all_params) if i % world_size == rank]
    other = [p for i, p in enumerate(all_params) if i % world_size != rank]
    return owned, other


def zero_memory_report(n_params: int, world_size: int = 1, stage: int = 0,
                       dtype_bytes: int = 2) -> Dict[str, float]:
    """估算各 ZeRO 阶段下**每卡**的状态显存（字节）。"""
    gb = 1024 ** 3
    param = n_params * dtype_bytes
    grad = n_params * dtype_bytes
    # fp32 主权重 + 一阶动量 + 二阶动量
    optim = n_params * 4 * 3
    if stage >= 3:
        per = (param + grad + optim) / world_size
    elif stage == 2:
        per = param + (grad + optim) / world_size
    elif stage == 1:
        per = param + grad + optim / world_size
    else:                       # stage 0 = 纯 DDP
        per = param + grad + optim
    return {
        "n_params": n_params,
        "no_sharding_gb": (param + grad + optim) / gb,
        "per_rank_gb": per / gb,
        "saved_ratio": 1.0 - per / (param + grad + optim),
    }


class ZeroOptimizer:
    """ZeRO-1 / 2 / 3 优化器封装。

    用法与 ``torch.optim.Optimizer`` 基本一致::

        zero = ZeroOptimizer(model.parameters(), ctx, stage=2, lr=1e-3)
        loss.backward()
        zero.step()
        zero.zero_grad()

    ``world_size=1`` 时与 AdamW 数值等价（这是我们的回归测试基准）。
    """

    def __init__(self, params: Iterable[nn.Parameter], ctx: Optional[DistContext] = None,
                 stage: int = 1, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        # weight_decay 默认与 torch.optim.AdamW 对齐（0.01），
        # 这样 world_size=1 时二者数值完全等价（有回归测试保证）。
        self.ctx = ctx or DistContext(world_size=1)
        if stage not in (0, 1, 2, 3):
            raise ValueError("stage 只能是 0/1/2/3")
        self.stage = stage
        self.world_size = self.ctx.world_size
        self.rank = self.ctx.rank
        self.params = [p for p in params if p.requires_grad]
        self.owned, self.other = partition_params(self.params, self.world_size, self.rank)
        self.optimizer = torch.optim.AdamW(
            self.owned, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.stats = {"steps": 0, "owned_params": len(self.owned),
                      "total_params": len(self.params)}

    # ------------------------------------------------------------------ #
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def _sync_grads(self) -> None:
        """stage 0/1：all-reduce 平均梯度；stage 2/3：reduce-scatter。"""
        if self.world_size == 1:
            return
        for p in self.params:
            if p.grad is None:
                continue
            if self.stage <= 1:
                g = self.ctx.all_reduce(p.grad)
                p.grad = g / self.world_size
            else:
                p.grad = self.ctx.reduce_scatter(p.grad, dim=0)

    @torch.no_grad()
    def _broadcast_params(self) -> None:
        """本 rank 只更新了自己那片参数，需要把新值同步给所有 rank。"""
        if self.world_size == 1:
            return
        for p in self.params:
            self.ctx.broadcast(p.data, src=(hash(id(p)) % self.world_size))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self) -> None:
        # stage 3 的前置动作：更新前把分片参数 all-gather 回来（由调用方在
        # forward 前后调用 gather_params/shard_params，这里只做一致性检查）
        if self.stage >= 3 and self.world_size > 1:
            raise RuntimeError("stage 3 需要配合 gather_params()/shard_params() 使用，见文档")

        self._sync_grads()
        self.optimizer.step()
        self._broadcast_params()
        self.stats["steps"] += 1

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def gather_params(self) -> None:
        """ZeRO-3：forward / backward 前把参数拼回完整（真实分布式下是 all-gather）。"""
        if self.stage < 3 or self.world_size == 1:
            return
        for p in self.params:
            if p.data.dim() == 0:
                continue
            p.data = self.ctx.all_gather(p.data, dim=0)

    @torch.no_grad()
    def shard_params(self) -> None:
        """ZeRO-3：用完即弃，只留本 rank 那一片。"""
        if self.stage < 3 or self.world_size == 1:
            return
        for p in self.params:
            if p.data.dim() == 0:
                continue
            p.data = p.data.chunk(self.world_size, dim=0)[self.rank].contiguous()

    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict:
        return {"stage": self.stage, "stats": dict(self.stats),
                "optimizer": self.optimizer.state_dict()}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ZeroOptimizer(stage={self.stage}, world_size={self.world_size}, "
                f"owned={self.stats['owned_params']}/{self.stats['total_params']})")
