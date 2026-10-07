"""线性注意力（RetNet 式 Retention）：O(L) 复杂度、O(1) 解码状态。

标准 softmax attention 的两个形态：
  * **并行形态**（训练 / prefill）：显式的 ``L×L`` 分数矩阵，O(L²)；
  * **递推形态**（decode）：维护一个 ``D×D`` 的状态矩阵 S，
    ``S_t = γ·S_{t-1} + k_t v_tᵀ``，``o_t = q_t S_t``，**每步 O(D²)**，
    与上下文长度无关，且不需要 KV Cache。

RetNet 的关键就是让**这两种形态数学等价**（用指数衰减 γ 代替 softmax），
于是训练可以像 Transformer 一样并行，推理却像 RNN 一样省。
代价：γ<1 的指数衰减是"固定偏置的局部性"，对超长精确召回弱于 softmax attention，
所以工业界普遍采用 ``window / linear / full`` 混合而不是纯线性。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .norms import RMSNorm

__all__ = ["LinearAttention"]


class LinearAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        H, D = cfg.n_heads, cfg.head_dim

        self.q_proj = nn.Linear(cfg.dim, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.dim, H * D, bias=False)
        self.v_proj = nn.Linear(cfg.dim, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.dim, bias=False)
        self.norm = RMSNorm(D)

        # 每个 head 一个衰减率 γ ∈ (0,1)，用 logit 参数化保证稳定
        self.log_decay = nn.Parameter(torch.log(torch.linspace(0.9, 0.99, H)))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.log_decay)          # [H]

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="linear") -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2) * (D ** -0.25)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2) * (D ** -0.25)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        if rope is not None:
            q, k = rope(q, k, positions)

        decode = (cache is not None and T == 1)
        if decode:
            out, state = self._recurrent(q, k, v, cache)
            cache.states[self.layer_idx] = state
        else:
            out = self._parallel(q, k, v)
            if cache is not None:
                # prefill 结束后把状态交给缓存，后续 decode 就能走 O(1) 递推
                cache.states[self.layer_idx] = self._init_state(k, v)

        out = self.norm(out.transpose(1, 2)).reshape(B, T, H * D)
        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    def _parallel(self, q, k, v) -> torch.Tensor:
        """训练 / prefill：显式 L×L 分数矩阵（带指数衰减与因果掩码）。"""
        B, H, T, D = q.shape
        g = self.gamma.view(1, H, 1, 1).to(q.dtype)
        idx = torch.arange(T, device=q.device)
        dist = idx.view(1, 1, T, 1) - idx.view(1, 1, 1, T)      # [1,1,T,T]
        decay = torch.where(dist >= 0, g ** dist.float(), torch.zeros((), dtype=q.dtype, device=q.device))
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * decay
        return torch.einsum("bhts,bhsd->bhtd", scores, v)

    # ------------------------------------------------------------------ #
    def _recurrent(self, q, k, v, cache):
        """decode：O(1) 状态递推。"""
        state = cache.states.get(self.layer_idx)
        if state is None:
            state = q.new_zeros(q.shape[0], self.n_heads, self.head_dim, self.head_dim)
        g = self.gamma.view(1, self.n_heads, 1, 1).to(state.dtype)
        state = g * state + torch.einsum("bhtd,bhte->bhde", k, v)
        out = torch.einsum("bhtd,bhde->bhte", q, state)
        return out, state

    def _init_state(self, k, v) -> torch.Tensor:
        """并行计算完一段后，把状态初始化为 Σ γ^(T-1-j) k_j v_jᵀ。"""
        B, H, T, D = k.shape
        g = self.gamma.view(1, H, 1, 1).to(k.dtype)
        idx = torch.arange(T, device=k.device).flip(0).view(1, 1, T, 1)
        decay = g ** idx.float()
        return torch.einsum("bhtd,bhte->bhde", k * decay, v)
