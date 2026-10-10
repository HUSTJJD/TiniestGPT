"""RWKV-7：无 Attention、无 KV Cache 的矩阵状态（广义 Delta Rule）。

与 GDN / Mamba 同属"固定状态"路线，但 RWKV-7 走得更远——
它**完全不保存**历史 token 的 K/V，每个 head 维护一个矩阵状态：

    S_t = S_{t-1} D_t                     # 逐通道时间衰减
        + (S_{t-1} a_t) b_t^T             # 基于旧状态的低秩擦除/改写
        + v_t k_t^T                       # 写入新的 key-value 关联
    o_t = S_t r_t

三项各司其职：衰减控制遗忘，低秩项负责**擦除**旧关联（这是 RWKV-7 相对
RWKV-6 的关键升级），第三项写入新关联。

状态复杂度::

    Attention:  O(L · H · d)        随上下文线性增长
    RWKV-7:     O(layers · H · d_k · d_v)   **与 L 无关**

所以 4K 与 1M token 的 decode 状态大小相同。

固定状态也有代价，必须写清楚：
  * 所有历史必须压进有限状态，精确逐 token 回看更难；
  * 不同请求无法像 prefix KV cache 那样共享任意前缀（要共享得存状态快照）；
  * beam search 每个 beam 都要复制/分叉状态；
  * 大 batch 时矩阵状态本身也吃显存；
  * 递推 decode 需要专用 kernel，不能复用 FlashAttention。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["RWKV7Mixer", "rwkv7_reference_scan"]


def rwkv7_reference_scan(r, k, v, w, a, b, state: torch.Tensor):
    """逐 token 的参考实现（**慢**，只用于数值对照）。

    输入均为 ``[B, H, T, D]``（w/a/b 为 ``[B, H, T, 1]``），``state`` 为 ``[B, H, Dv, Dk]``。
    """
    B, H, T, D = k.shape
    outs = []
    S = state
    for t in range(T):
        kt, vt = k[:, :, t], v[:, :, t]           # [B,H,D]
        rt, wt = r[:, :, t], w[:, :, t]           # [B,H,D] / [B,H,1]
        at, bt = a[:, :, t], b[:, :, t]           # [B,H,D]
        # S ← S · diag(w) − (S·(k·a)) k^T·b ... 用官方等价形式写：
        decay = torch.exp(-F.softplus(wt)).unsqueeze(-1)          # [B,H,D,1]
        S = S * decay.transpose(-1, -2)                            # 逐列衰减
        ka = (kt * at).unsqueeze(-1)                               # [B,H,D,1]
        S = S - torch.matmul(S, ka) * (kt * bt).unsqueeze(-2)      # 低秩擦除
        S = S + vt.unsqueeze(-1) * kt.unsqueeze(-2)                # 写入新关联
        outs.append(torch.matmul(S, rt.unsqueeze(-1)).squeeze(-1))  # [B,H,D]
    return torch.stack(outs, dim=2), S


class RWKV7Mixer(nn.Module):
    """RWKV-7 的序列混合器（接口与其它 mixer 一致）。"""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, D = cfg.n_heads, cfg.head_dim
        self.n_heads, self.head_dim = H, D
        self.state_shape = (H, D, D)

        self.receptance = nn.Linear(cfg.dim, H * D, bias=False)
        self.key = nn.Linear(cfg.dim, H * D, bias=False)
        self.value = nn.Linear(cfg.dim, H * D, bias=False)
        self.decay = nn.Linear(cfg.dim, H, bias=False)         # 每头一个通道缩放
        self.erase_a = nn.Linear(cfg.dim, H * D, bias=False)
        self.erase_b = nn.Linear(cfg.dim, H * D, bias=False)
        self.out_proj = nn.Linear(H * D, cfg.dim, bias=False)
        self.ln_x = nn.LayerNorm(H * D)

    # ------------------------------------------------------------------ #
    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _new_state(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, D = self.n_heads, self.head_dim
        return torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="full") -> torch.Tensor:
        B, T, _ = x.shape
        r = self._split(self.receptance(x))
        k = self._split(self.key(x))
        v = self._split(self.value(x))
        w = self.decay(x).transpose(1, 2).unsqueeze(-1)         # [B,H,T,1]
        a = self._split(self.erase_a(x))
        b = self._split(self.erase_b(x))

        state = cache.states.get(self.layer_idx) if cache is not None else None
        if state is None:
            state = self._new_state(x)

        out, state = rwkv7_reference_scan(r, k, v, w, a, b, state)
        if cache is not None:
            cache.states[self.layer_idx] = state

        y = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(self.ln_x(y))
