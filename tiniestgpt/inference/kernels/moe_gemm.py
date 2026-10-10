"""MoE 推理内核：分组 GEMM + Expert Parallel + 融合 permute。

朴素实现（``moe.py`` 里那样）有三个性能陷阱：

1. **逐专家 Python 循环**：expert 数一多（256/384），循环开销主导，
   而且每个 expert 拿到的是一个**窄矩阵**，Tensor Core 利用率极差；
2. **token 分发/聚合**（permute / unpermute）是两次完整的读写，
   在生产实现里必须和 All-to-All 融合；
3. **负载不均**：某个 expert 拿到 3 倍 token，整层就得等它。

本模块给出：

* :func:`grouped_gemm` —— 一次调用里跑完所有 expert 的 GEMM（按 expert 分组的 batched 乘）；
* :func:`fused_permute` / :func:`fused_unpermute` —— 排序与还原；
* :class:`ExpertParallel` —— 模拟 EP 的 All-to-All 与通信量统计；
* :func:`load_balance_report` —— 把负载不均衡量化出来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["fused_permute", "fused_unpermute", "grouped_gemm", "ExpertParallel",
           "load_balance_report", "moe_grouped_forward"]


def fused_permute(x: torch.Tensor, topk_idx: torch.Tensor, n_experts: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 token 按 expert 排序（一次 argsort，后续所有 expert 共用）。

    :return: ``(sorted_x, sorted_idx, expert_offsets, src_positions)``
    """
    N, K = topk_idx.shape
    flat = topk_idx.reshape(-1)                       # [N*K]
    order = torch.argsort(flat, stable=True)          # 按 expert id 排序
    dst_token = order // K                            # 每个 (token,slot) 对应的 token
    sorted_x = x.index_select(0, dst_token)
    counts = torch.bincount(flat, minlength=n_experts)
    offsets = torch.zeros(n_experts + 1, dtype=torch.long, device=x.device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return sorted_x, flat[order], offsets, order


def fused_unpermute(y_sorted: torch.Tensor, order: torch.Tensor,
                    topk_idx: torch.Tensor, weights: torch.Tensor,
                    N: int, dim: int) -> torch.Tensor:
    """把排序后的输出还原回 token 顺序，并按路由权重加权求和。"""
    K = topk_idx.shape[1]
    w = weights.reshape(-1).index_select(0, order).to(y_sorted.dtype)
    out = torch.zeros(N, dim, device=y_sorted.device, dtype=y_sorted.dtype)
    contrib = y_sorted * w.unsqueeze(-1)
    dst = order // K
    out.index_add_(0, dst, contrib)
    return out


def grouped_gemm(sorted_x: torch.Tensor, weight: torch.Tensor,
                 offsets: torch.Tensor) -> torch.Tensor:
    """分组 GEMM：``weight`` 为 ``[E, out, in]``，按 offsets 给每段选对应 expert。

    教学实现用**分段循环**代替真正的 grouped GEMM kernel，
    但**接口与语义**和生产实现一致，可以直接换成 Triton/CUTLASS 的 grouped GEMM。
    """
    E = weight.shape[0]
    outs: List[torch.Tensor] = []
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if hi <= lo:
            continue
        outs.append(F.linear(sorted_x[lo:hi], weight[e]))
    return torch.cat(outs, dim=0) if outs else sorted_x.new_zeros(0, weight.shape[1])


def moe_grouped_forward(x: torch.Tensor, gate_w: torch.Tensor,
                        w1: torch.Tensor, w2: torch.Tensor,
                        top_k: int = 2, score_func: str = "sigmoid"
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """一次完整的分组 MoE 前向（不含共享专家）。返回 ``(out, topk_idx)``。"""
    N, C = x.shape
    E = w1.shape[0]
    logits = F.linear(x, gate_w).float()
    probs = torch.sigmoid(logits) if score_func == "sigmoid" else torch.softmax(logits, dim=-1)
    weights, topk_idx = torch.topk(probs, min(top_k, E), dim=-1)

    sx, _, offsets, order = fused_permute(x, topk_idx, E)
    h = grouped_gemm(sx, w1, offsets)
    h = F.silu(h)
    y = grouped_gemm(h, w2, offsets)
    out = fused_unpermute(y, order, topk_idx, weights, N, C)
    return out, topk_idx


@dataclass
class ExpertParallel:
    """Expert Parallel 的通信模型（单进程模拟）。

    EP 的关键代价是 **All-to-All**：每张卡把自己这部分的 token 发到
    拥有目标 expert 的卡上，算完再发回来。payload 与「被路由的向量宽度 × 份数」成正比——
    这正是 LatentMoE 要先降维再路由的原因。
    """

    world_size: int = 1
    comm_bytes: int = 0
    dispatches: int = 0

    def all_to_all(self, tokens: int, hidden: int, top_k: int,
                   dtype_bytes: int = 2) -> int:
        """返回本次 dispatch+combine 的通信字节数（双向）。"""
        payload = tokens * hidden * top_k * dtype_bytes
        self.comm_bytes += 2 * payload          # 去 + 回
        self.dispatches += 1
        return 2 * payload

    def report(self) -> str:
        return (f"ExpertParallel(world={self.world_size}): {self.dispatches} 次 dispatch, "
                f"通信 {self.comm_bytes / 1e6:.1f} MB")


def load_balance_report(topk_idx: torch.Tensor, n_experts: int) -> Dict[str, float]:
    """把专家负载不均衡量化出来（这是 MoE 训练/推理的第一监控项）。"""
    flat = topk_idx.reshape(-1)
    counts = torch.bincount(flat, minlength=n_experts).float()
    mean = counts.mean().clamp(min=1e-9)
    return {
        "n_experts": float(n_experts),
        "mean_load": float(mean),
        "max_load": float(counts.max()),
        "min_load": float(counts.min()),
        "imbalance": float(counts.max() / mean),      # 越接近 1 越好
        "cv": float(counts.std() / mean),             # 变异系数
        "dead_experts": float((counts == 0).sum()),   # 饿死的专家数
    }
