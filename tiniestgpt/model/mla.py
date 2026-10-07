"""MLA（Multi-head Latent Attention，DeepSeek V2/V3）：**低秩 KV 压缩**。

思路：
  1. 把每个 token 的 K/V 压成一条维度为 ``d_c`` 的**潜向量** c（而不是 2·H·D）；
  2. 推理时**只缓存这条潜向量**，KV Cache 体积直接降到约 ``d_c / (2·H·D)``；
  3. 位置信息用一条**解耦的 RoPE 向量**（所有 head 共享）单独承载，
     避免"压缩后再旋转"破坏相对位置的线性可加性；
  4. 矩阵吸收技巧（absorbed form）可把 up-projection 融进 Q/O 投影，
     decode 时等价于一次更大的 GEMM（本项目用显式形式，便于对照理解）。

代价：Q 也需要拆成 nope/rope 两段，且 head_dim 被切成 ``(D - d_rope) + d_rope``。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .dispatch import get_dense_attention, get_paged_attention
from .norms import QKNorm

__all__ = ["MultiHeadLatentAttention"]


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, D = cfg.n_heads, cfg.head_dim
        self.n_heads = H
        self.head_dim = D
        self.latent = cfg.mla_latent_dim
        self.rope_dim = min(cfg.mla_rope_dim, D - 1)
        self.content_dim = D - self.rope_dim
        self.softcap = cfg.attn_softcap
        self.attn_backend = cfg.attn_backend

        # Q：内容部分（每头独立）+ 共享的 RoPE 部分
        self.q_proj = nn.Linear(cfg.dim, H * self.content_dim, bias=False)
        self.q_rope_proj = nn.Linear(cfg.dim, self.rope_dim, bias=False)

        # KV：下投影到潜向量 + 共享 RoPE 通路
        self.kv_down = nn.Linear(cfg.dim, self.latent + self.rope_dim, bias=False)
        self.k_up = nn.Linear(self.latent, H * self.content_dim, bias=False)
        self.v_up = nn.Linear(self.latent, H * self.content_dim, bias=False)
        # 注意：attention 输出的 head 维度等于 **V 的维度**（content_dim），
        # 不含解耦的 RoPE 部分——这是 MLA 与标准 MHA 在形状上最容易搞错的地方。
        self.o_proj = nn.Linear(H * self.content_dim, cfg.dim, bias=False)

        self.q_norm = QKNorm(D) if cfg.qk_norm else None

    # ------------------------------------------------------------------ #
    @property
    def cache_dim(self) -> int:
        """每层每 token 需要缓存的浮点数个数（LM 压缩后的 KV 体积）。"""
        return self.latent + self.rope_dim

    def _rope_apply(self, q_rope: torch.Tensor, k_rope: torch.Tensor,
                    rope, positions) -> tuple[torch.Tensor, torch.Tensor]:
        """对共享的 RoPE 通路施加旋转（q_rope/k_rope: [B,T,1,r]）。"""
        if rope is None:
            return q_rope, k_rope
        q4 = q_rope.transpose(1, 2)   # [B,1,T,r]
        k4 = k_rope.transpose(1, 2)
        q4, k4 = rope(q4, k4, positions)
        return q4.transpose(1, 2), k4.transpose(1, 2)

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="full") -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        r = self.rope_dim

        # --- Q ---
        q_nope = self.q_proj(x).view(B, T, H, self.content_dim)         # [B,T,H,dc]
        q_rope = self.q_rope_proj(x).view(B, T, 1, r)                    # [B,T,1,r]
        # --- KV 潜向量 ---
        kv = self.kv_down(x)
        c = kv[..., :self.latent]                                        # [B,T,dc_latent]
        k_rope = kv[..., self.latent:].view(B, T, 1, r)                  # [B,T,1,r]
        q_rope, k_rope = self._rope_apply(q_rope, k_rope, rope, positions)
        q_rope = q_rope.expand(B, T, H, r)
        q = torch.cat([q_nope, q_rope], dim=-1)                          # [B,T,H,D]
        if self.q_norm is not None:
            q = self.q_norm(q)

        latent = torch.cat([c, k_rope.squeeze(2)], dim=-1)               # [B,T,latent+r]

        # --- 缓存：只存潜向量 ---
        if cache is not None:
            view = cache.write(self.layer_idx, latent.unsqueeze(2), None)  # [B,T,1,L]
            if view.kind == "paged":
                c_all, kr_all, kv_len = self._read_paged(view)
            else:
                lat = view.k                                             # [B,Tk,1,L]
                c_all = lat[..., 0, :self.latent]
                kr_all = lat[..., 0, self.latent:]
                kv_len = view.kv_len
        else:
            c_all, kr_all, kv_len = c, k_rope.squeeze(2), T

        # --- 上投影还原 K/V ---
        k = self.k_up(c_all).view(B, kv_len, H, self.content_dim)
        v = self.v_up(c_all).view(B, kv_len, H, self.content_dim)
        kr = kr_all.view(B, kv_len, 1, r).expand(B, kv_len, H, r)
        k = torch.cat([k, kr], dim=-1)                                   # [B,Tk,H,D]

        q4 = q.transpose(1, 2)                                           # [B,H,T,D]
        k4 = k.transpose(1, 2)
        v4 = v.transpose(1, 2)

        window = self.cfg.attn_window if (layer_type == "window" and self.cfg.attn_window > 0) else -1
        sinks = self.cfg.attn_sinks if window > 0 else 0

        if cache is not None and hasattr(view, "kind") and view.kind == "paged":
            # 已还原成稠密 K/V，直接走稠密后端即可（MLA 的 KV 已在读回时展开）
            fn = get_dense_attention(self.attn_backend)
            out = fn(q4, k4, v4, mask=None, is_causal=is_causal,
                     window=window, sinks=sinks, offset=kv_len - T, softcap=self.softcap)
        else:
            fn = get_dense_attention(self.attn_backend)
            out = fn(q4, k4, v4, mask=attn_mask, is_causal=is_causal,
                     window=window, sinks=sinks, offset=kv_len - T, softcap=self.softcap)

        out = out.transpose(1, 2).reshape(B, T, H * self.content_dim)
        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    def _read_paged(self, view):
        """从分页缓存中 gather 出每个序列的潜向量（padded→dense）。"""
        block_table = view.block_table
        seq_lens = view.seq_lens
        k_cache = view.k
        B = block_table.shape[0]
        L = int(seq_lens.max().item())
        lat_dim = k_cache.shape[-1]
        device = k_cache.device
        out = torch.zeros(B, L, lat_dim, device=device, dtype=k_cache.dtype)
        for b in range(B):
            n = int(seq_lens[b])
            blocks = block_table[b][block_table[b] >= 0]
            flat = k_cache[blocks].reshape(-1, lat_dim)[:n]
            out[b, :n] = flat
        c_all = out[..., :self.latent]
        kr_all = out[..., self.latent:]
        return c_all, kr_all, L
