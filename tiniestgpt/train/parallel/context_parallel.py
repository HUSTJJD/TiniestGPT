"""Context Parallel / Ring Attention：把**序列维度**切到多张卡上。

TP 切的是 hidden、PP 切的是 layer、ZeRO 切的是 optimizer state，
而 1M 上下文训练的瓶颈在 attention 的 O(L²) 与 activation——
这两者都要靠**切序列**来解决。

Ring Attention 的做法：

1. 每张卡持有自己那一段 Q；
2. K/V 在卡之间**环形传递**一圈，每张卡每次拿到别人的一段 K/V 做局部 attention；
3. 用 **online softmax**（把每段的部分和/最大值/m 归一化状态一起传）把各段结果合并，
   得到与"一次性看全序列"**数值等价**的结果。

关键性质：通信量是 O(L) 而不是 O(L²)，且不需要物化 [L, L] 的 score 矩阵。

与其它 parallel 模块一致，这里提供**单进程模拟**：
把序列切成 ``world_size`` 段，按环形顺序累加，验证与全量 attention 等价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["ContextParallelConfig", "ring_attention", "split_sequence",
           "OnlineSoftmaxState", "context_parallel_attention"]


@dataclass
class ContextParallelConfig:
    world_size: int = 2
    causal: bool = True
    ring: bool = True            # False = 退化为"各算各的"（用于对照，结果会不同）


class OnlineSoftmaxState:
    """online softmax 的累加状态：``(running_max, running_sum, accumulator)``。

    合并两段时先对齐指数基准，再相加——这是 flash attention 的标准技巧，
    也是 ring attention 能分段计算的根本原因。
    """

    def __init__(self, shape: Tuple[int, int, int], head_dim: int,
                 device, dtype) -> None:
        """``shape = (B, H, T)``。``m/l`` 为 [B,H,T]，``acc`` 为 [B,H,T,D]。"""
        self.m = torch.full(shape, float("-inf"), device=device, dtype=dtype)
        self.l = torch.zeros(shape, device=device, dtype=dtype)
        self.acc = torch.zeros((*shape, head_dim), device=device, dtype=dtype)

    def update(self, s: torch.Tensor, v: torch.Tensor) -> None:
        """s:[B,H,T,K]（未归一化 score）  v:[B,H,K,D]"""
        new_m = torch.maximum(self.m, s.amax(dim=-1))
        p = torch.exp(s - new_m[..., None])
        corr = torch.exp(self.m - new_m)                  # 把旧的 max 换算到新的基准
        self.l = self.l * corr + p.sum(dim=-1)
        self.acc = self.acc * corr[..., None] + torch.matmul(p, v)
        self.m = new_m

    def output(self) -> torch.Tensor:
        return self.acc / self.l.clamp(min=1e-12).unsqueeze(-1)


def split_sequence(t: torch.Tensor, world_size: int) -> List[torch.Tensor]:
    """把 [B, L, H, D] 沿序列维切成 world_size 段（不足则最后一段为空）。"""
    L = t.shape[1]
    per = (L + world_size - 1) // world_size
    return [t[:, r * per:(r + 1) * per] for r in range(world_size)]


def ring_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   cfg: ContextParallelConfig) -> torch.Tensor:
    """分块环形注意力，返回与全量 causal attention **等价**的结果。

    q/k/v 形状均为 ``[B, L, H, D]``。
    """
    B, L, H, D = q.shape
    ws = cfg.world_size
    q_chunks = split_sequence(q, ws)
    k_chunks = split_sequence(k, ws)
    v_chunks = split_sequence(v, ws)
    scale = D ** 0.5

    outs: List[torch.Tensor] = []
    per = (L + ws - 1) // ws
    for r in range(ws):
        qc = q_chunks[r]                     # 本 rank 的 query 段
        if qc.shape[1] == 0:
            continue
        # 每个 query 的绝对起始位置
        base = r * per
        st = OnlineSoftmaxState((B, H, qc.shape[1]), D, qc.device, qc.dtype)
        for step in range(ws):
            src = (r - step) % ws            # 环形：优先看自己，再依次看前面的段
            kc, vc = k_chunks[src], v_chunks[src]
            if kc.shape[1] == 0:
                continue
            s = torch.einsum("blhd,bshd->bhls", qc, kc) / scale
            if cfg.causal:
                q_abs = base + torch.arange(qc.shape[1], device=qc.device)
                k_abs = src * per + torch.arange(kc.shape[1], device=kc.device)
                s = s.masked_fill(k_abs.view(1, 1, 1, -1) > q_abs.view(1, 1, -1, 1),
                                  float("-inf"))
            st.update(s, vc.transpose(1, 2))     # v → [B,H,S,D]
        outs.append(st.output().transpose(1, 2))  # [B, T_r, H, D]
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def context_parallel_attention(q, k, v, cfg: ContextParallelConfig) -> torch.Tensor:
    """对外入口：等价于 ``softmax(qkᵀ/√d) v`` 的因果注意力，但分块计算。"""
    if cfg.world_size <= 1:
        s = torch.einsum("blhd,bshd->bhls", q, k) / (q.shape[-1] ** 0.5)
        if cfg.causal:
            L = q.shape[1]
            mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, v.transpose(1, 2)).transpose(1, 2)
    return ring_attention(q, k, v, cfg)
