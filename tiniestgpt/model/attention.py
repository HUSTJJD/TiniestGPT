"""注意力：MHA / GQA / MQA 的统一实现 + 滑动窗口 + Attention Sink + 可插拔后端。

为什么这些结构会被发明（一句话总结）：
  * **GQA/MQA**：decode 的瓶颈是**读 KV 的带宽**。把 K/V 头数从 H 降到 g（甚至 1），
    显存与带宽直接除以 H/g，质量损失极小 → 几乎所有现代模型都用它。
  * **滑动窗口**（Mistral/Longformer）：局部性假设——绝大多数依赖都在附近，
    窗口外靠"层间传递"，把长上下文的注意力成本从 O(L²) 降到 O(L·W)。
  * **Attention Sink**（StreamingLLM）：前几个 token（通常是 BOS）会被**所有**
    位置强烈关注；一旦把它们挤出窗口，模型会瞬间崩溃。因此必须**永久保留**。
  * **logit soft-capping**（Gemma2）：用 ``tanh(x/c)·c`` 压住极端 logits，
    比硬性裁剪更平滑，是低精度训练稳定性的关键技巧。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .dispatch import get_dense_attention, get_paged_attention
from .kv_cache import CacheView
from .norms import QKNorm

__all__ = ["Attention"]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_groups = cfg.n_groups
        self.attn_backend = cfg.attn_backend

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        # K=V 共享（DeepSeek-V4 long-range MQA / Gemma 4 global）：
        # 只缓存一份，K 与 V 取同一表示；省一半 KV Cache 与写入带宽。
        self.kv_eq_v = bool(getattr(cfg, "kv_eq_v", False))
        self.v_proj = None if self.kv_eq_v else \
            nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        # 超长上下文推理的 attention 温度缩放（Llama 4 Scout 的做法）
        self.temperature = float(getattr(cfg, "attn_temperature", 0.0) or 0.0)

        self.q_norm = QKNorm(cfg.head_dim) if cfg.qk_norm else None
        self.k_norm = QKNorm(cfg.head_dim) if cfg.qk_norm else None
        self.softcap = cfg.attn_softcap

    # ------------------------------------------------------------------ #
    @property
    def window(self) -> int:
        return self.cfg.attn_window

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, C]
        positions: Optional[torch.Tensor] = None,
        rope=None,
        cache=None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        layer_type: str = "full",
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)      # [B,H,T,D]
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        # K=V 时 V 直接取 K 的表示（省掉一份投影与一份缓存）
        v = k if self.kv_eq_v else self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        if self.temperature > 0:
            q = q / self.temperature

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = rope(q, k, positions)

        window = self.window if (layer_type == "window" and self.window > 0) else -1
        sinks = self.cfg.attn_sinks if window > 0 else 0

        # ---------------- 写入 / 读取 KV Cache ----------------
        if cache is not None:
            view: CacheView = cache.write(self.layer_idx, k.transpose(1, 2), v.transpose(1, 2))
            if view.kind == "paged_quant":
                from .kernels import paged_attention_quantized_ref

                out = paged_attention_quantized_ref(
                    q, view.k, view.v, view.k_scale, view.v_scale,
                    view.block_table, view.seq_lens, causal=is_causal,
                    window=window, sinks=sinks)
            elif view.kind == "paged":
                fn = get_paged_attention(self.attn_backend)
                out = fn(q, view.k, view.v, view.block_table, view.seq_lens,
                         causal=is_causal, window=window, sinks=sinks,
                         softcap=self.softcap, max_len=view.max_len)
            else:
                k = view.k.transpose(1, 2) if view.k is not None else k
                v = view.v.transpose(1, 2) if view.v is not None else v
                offset = view.kv_len - T
                fn = get_dense_attention(self.attn_backend)
                out = fn(q, k, v, mask=attn_mask, is_causal=is_causal,
                         window=window, sinks=sinks, offset=offset, softcap=self.softcap)
        else:
            fn = get_dense_attention(self.attn_backend)
            out = fn(q, k, v, mask=attn_mask, is_causal=is_causal,
                     window=window, sinks=sinks, offset=0, softcap=self.softcap)

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(out)
