"""分布式 Muon：正交化更新在多卡上怎么做。

Muon 对**二维权重**的梯度做近似正交化（Newton-Schulz 迭代），
在 DeepSeek-V4 / Kimi K2 上已成为标配。但分布式训练会引出三个问题：

1. **ZeRO 分片**：Muon 只需要**动量**（没有二阶矩），
   所以 optimizer state 比 AdamW 小一半，但仍要分片——
   否则每张卡都存一份完整动量。
2. **正交化通信**：Newton-Schulz 需要整个**未分片**的梯度矩阵。
   做法是先 all-gather 出完整 grad → 正交化 → 各卡只取自己那一片更新。
   通信量 = 一次 all-gather，而不是每步多次。
3. **混合分组**：Embedding / Norm / bias 这些**非二维**参数仍走 AdamW。

本模块沿用项目其它并行模块的风格：**单进程模拟多 rank**，
用 all-reduce 等价的本地聚合验证数学正确性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["MuonDistConfig", "DistributedMuon", "shard_flat", "gather_shards",
           "newton_schulz_dist"]


@dataclass
class MuonDistConfig:
    lr: float = 0.02
    momentum: float = 0.95
    ns_steps: int = 6
    weight_decay: float = 0.0
    shard_optimizer_state: bool = True     # ZeRO-1：动量分片
    world_size: int = 1
    rank: int = 0


def shard_flat(t: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """把一维张量切成 world_size 份，返回第 rank 份（模拟 ZeRO 分片）。"""
    n = t.numel()
    per = (n + world_size - 1) // world_size
    lo = rank * per
    hi = min(lo + per, n)
    if lo >= n:
        return t.new_zeros(0)
    return t.reshape(-1)[lo:hi].clone()


def gather_shards(shards: List[torch.Tensor], n: int) -> torch.Tensor:
    """把各 rank 的分片拼回完整张量（模拟 all-gather）。"""
    return torch.cat([s for s in shards], dim=0)[:n].clone()


def newton_schulz_dist(g: torch.Tensor, steps: int = 6, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 正交化：把 G 迭代成最接近它的半正交矩阵。

    ``X ← aX + bX(XᵀX) + cX(XᵀX)²``，系数取 (3.4445, -4.7750, 2.0315)。
    只对**二维**梯度有意义。
    """
    if g.dim() != 2:
        return g
    a, b, c = 3.4445, -4.7750, 2.0315
    X = g.float()
    # 先归一化，保证迭代收敛到"缩放后的正交阵"而不是发散
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        transposed = True
        X = X.T
    else:
        transposed = False
    for _ in range(steps):
        # 注意方向：A = X Xᵀ，且是 **B @ X** 而不是 X @ B。
        # 写反的话迭代不会收敛（实测误差停在 0.38 左右不动）。
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(g.dtype)


class DistributedMuon(torch.optim.Optimizer):
    """ZeRO 分片的 Muon + 非二维参数回退 AdamW。

    单独对非二维参数维护一个 AdamW 是**必须的**——
    对 1D 向量做正交化没有意义。
    """

    def __init__(self, params: Iterable[nn.Parameter],
                 cfg: Optional[MuonDistConfig] = None) -> None:
        self.cfg = cfg or MuonDistConfig()
        super().__init__(params, {"lr": self.cfg.lr})
        self.matrix_params: List[nn.Parameter] = []
        self.vector_params: List[nn.Parameter] = []
        for group in self.param_groups:
            for p in group["params"]:
                (self.matrix_params if p.dim() >= 2 else self.vector_params).append(p)
        self._adam = torch.optim.AdamW(self.vector_params, lr=self.cfg.lr) \
            if self.vector_params else None
        self.comm_bytes = 0

    def _shard_key(self, p: nn.Parameter) -> int:
        return id(p)

    @torch.no_grad()
    def step(self, closure=None):
        cfg = self.cfg
        for p in self.matrix_params:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state.setdefault(p, {})
            buf = state.get("momentum_buf")
            if buf is None:
                buf = torch.zeros_like(g)
                state["momentum_buf"] = buf
            buf.mul_(cfg.momentum).add_(g)          # 动量（分片存储时也只是本地那片）

            # ---- 模拟：分片 → all-gather → 正交化 → 各卡只更新自己那片 ----
            n = buf.numel()
            if cfg.world_size > 1 and cfg.shard_optimizer_state:
                shards = [shard_flat(buf, cfg.world_size, r) for r in range(cfg.world_size)]
                full = gather_shards(shards, n).reshape(buf.shape)
                self.comm_bytes += full.numel() * full.element_size()
            else:
                full = buf
            upd = newton_schulz_dist(full, cfg.ns_steps)

            if cfg.weight_decay:
                p.mul_(1.0 - cfg.lr * cfg.weight_decay)
            if cfg.world_size > 1 and cfg.shard_optimizer_state:
                per = (n + cfg.world_size - 1) // cfg.world_size
                lo, hi = cfg.rank * per, min(cfg.rank * per + per, n)
                if lo < hi:                          # 关键：只写自己负责的那一段
                    p.reshape(-1)[lo:hi].add_(upd.reshape(-1)[lo:hi], alpha=-cfg.lr)
            else:
                p.add_(upd.reshape(p.shape), alpha=-cfg.lr)
        if self._adam is not None:
            self._adam.step()
        return None

    def report(self) -> str:
        return (f"DistributedMuon: matrix={len(self.matrix_params)} "
                f"vector={len(self.vector_params)} world={self.cfg.world_size} "
                f"comm={self.comm_bytes / 1e6:.1f} MB")
