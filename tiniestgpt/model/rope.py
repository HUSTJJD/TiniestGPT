"""旋转位置编码（RoPE）家族：RoPE / YaRN / mRoPE / 部分旋转。

要点回顾：
  * **RoPE 的本质**是把 Q/K 的每两个维度看成复数，乘上 ``e^{i·m·θ}``。
    这样点积只依赖**相对位置** m-n，天然支持外推（但原始 RoPE 外推能力有限）。
  * **YaRN** 对"低频分量"做插值、对"高频分量"保持原样（用 ramp mask 平滑过渡），
    再叠加一个 attention scale 修正，用极少的长文本数据即可扩展上下文。
  * **部分旋转**（DeepSeek / MLA）只让一部分维度参与旋转，留一部分给
    "无位置信息的语义通道"，配合低秩 KV 压缩效果更好。
  * **mRoPE**（多模态）把维度切成若干段，每段用不同的位置索引（t/h/w）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["RotaryEmbedding", "apply_rotary_emb", "yarn_inv_freq", "yarn_attention_scale"]


def yarn_find_correction_dim(num_rot: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def yarn_linear_ramp_mask(low: float, high: float, dim: int, device=None) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def yarn_inv_freq(dim: int, base: float, factor: float, original_max_len: int,
                  beta_fast: int = 32, beta_slow: int = 1, device=None) -> torch.Tensor:
    """YaRN 的 inverse frequency：低频插值、高频保留。"""
    pos = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq_extra = 1.0 / (base ** (pos / dim))
    if factor <= 1.0:
        return inv_freq_extra
    inv_freq_inter = 1.0 / (factor * base ** (pos / dim))
    low = yarn_find_correction_dim(beta_fast, dim, base, original_max_len)
    high = yarn_find_correction_dim(beta_slow, dim, base, original_max_len)
    ramp = yarn_linear_ramp_mask(low, high, dim // 2, device=device)
    return inv_freq_inter * (1.0 - ramp) + inv_freq_extra * ramp


def yarn_get_mscale(factor: float, mscale: float = 0.1) -> float:
    """HF 实现中的 ``yarn_get_mscale``：``mscale · ln(s) + 1``。"""
    if factor <= 1.0:
        return 1.0
    return float(mscale * math.log(factor) + 1.0)


def yarn_attention_scale(factor: float, mscale: float = 0.1) -> float:
    """YaRN 对 attention logits 的缩放系数 ``(0.1·ln s + 1)²``。

    实现方式是把 cos/sin 同时乘上 ``sqrt(scale)``，从而让 q·k 整体乘上该系数。
    """
    m = yarn_get_mscale(factor, mscale)
    return m * m


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把 (a, b) 变成 (-b, a)：复数乘 i。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x 的前 cos.shape[-1] 维做旋转，其余维原样保留（部分旋转）。

    :param x:   [..., D]
    :param cos: [..., D_rot]，会广播到 x 的除最后一维外所有维
    """
    d_rot = cos.shape[-1]
    x_rot = x[..., :d_rot]
    x_pass = x[..., d_rot:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


class RotaryEmbedding(nn.Module):
    """预计算并缓存 cos/sin；forward 时按 position_ids 查表。"""

    def __init__(self, head_dim: int, rope_type: str = "rope", theta: float = 10000.0,
                 scaling: float = 1.0, original_max_len: int = 2048,
                 max_seq_len: int = 2048, partial: float = 1.0,
                 mscale: float = 1.0, mrope_sections: int = 3) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_type = rope_type.lower()
        self.partial = float(partial)
        self.mscale = mscale
        self.scaling = scaling
        self.mrope_sections = mrope_sections

        # 参与旋转的维度必须是偶数
        d = int(head_dim * self.partial)
        d = d - (d % 2)
        self.rot_dim = max(d, 0)

        if self.rot_dim > 0 and self.rope_type != "none":
            if self.rope_type == "yarn":
                inv_freq = yarn_inv_freq(self.rot_dim, theta, scaling, original_max_len)
            else:
                inv_freq = 1.0 / (theta ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._build_cache(max_seq_len)
        else:
            self.register_buffer("inv_freq", torch.zeros(0), persistent=False)
            self.register_buffer("cos_cached", torch.zeros(0), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(0), persistent=False)

        if self.rope_type == "yarn" and scaling > 1.0:
            self.attn_scale = yarn_attention_scale(scaling, mscale)
        else:
            self.attn_scale = 1.0

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_cache(self, max_seq_len: int) -> None:
        if self.rot_dim == 0:
            return
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)              # [T, rot_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)            # [T, rot_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_cached = max_seq_len

    # ------------------------------------------------------------------ #
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                positions: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 q/k 施加旋转。

        :param q/k: [B, H, T, D]
        :param positions: [B, T] 或 [B, T, S]（mRoPE）；None 则用 0..T-1
        """
        if self.rot_dim == 0 or self.rope_type == "none":
            return q, k

        T = q.shape[2]
        device = q.device
        if positions is None:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(q.shape[0], T)

        if positions.dim() == 3:
            # mRoPE：不同维度段使用不同的位置索引
            return self._mrope(q, k, positions)

        need = int(positions.max().item()) + 1 if positions.numel() else 0
        if need > self.cos_cached.shape[0]:
            self._build_cache(max(need + 128, int(self.cos_cached.shape[0] * 2)))
        cos = self.cos_cached[positions].to(device=device, dtype=q.dtype)   # [B, T, rot_dim]
        sin = self.sin_cached[positions].to(device=device, dtype=q.dtype)
        # 输入的最后一维可能小于 rot_dim（例如 MLA 只给一部分维度做旋转）
        d_eff = min(cos.shape[-1], q.shape[-1])
        cos = cos[..., :d_eff]
        sin = sin[..., :d_eff]
        if self.attn_scale != 1.0:
            # YaRN：把 scale 平分到 cos/sin 上，等价于给 q·k 乘上 attn_scale
            s = math.sqrt(self.attn_scale)
            cos = cos * s
            sin = sin * s
        cos = cos.unsqueeze(1)     # [B, 1, T, rot_dim]
        sin = sin.unsqueeze(1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    # ------------------------------------------------------------------ #
    def _mrope(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T, D = q.shape
        S = positions.shape[-1]
        if self.rot_dim % S != 0:
            raise ValueError("mRoPE 要求 rot_dim 能被段数整除")
        sec = self.rot_dim // S
        q_out = q.clone()
        k_out = k.clone()
        for s in range(S):
            pos = positions[..., s]                            # [B, T]
            cos = self.cos_cached[pos][..., :sec].unsqueeze(1)  # [B,1,T,sec]
            sin = self.sin_cached[pos][..., :sec].unsqueeze(1)
            lo, hi = s * sec, (s + 1) * sec
            q_out[..., lo:hi] = apply_rotary_emb(q[..., lo:hi], cos, sin)
            k_out[..., lo:hi] = apply_rotary_emb(k[..., lo:hi], cos, sin)
        return q_out, k_out
