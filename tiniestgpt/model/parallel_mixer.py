"""并行 Attention-SSM 混合块（Falcon-H1 路线）。

Nemotron / Qwen3.6 的混合是**按层交替**：

    Mamba → MoE → Mamba → MoE → Attention → MoE → ...

Falcon-H1 是**同一块内并行**，两条分支各自占一部分通道::

                     +-> Attention heads --+
    x -> Norm -> proj|                     |-> concat -> output proj
                     +-> Mamba-2 heads ----+

    a = Attention(P_A x)
    s = Mamba2(P_S x)
    y = W_O concat(a, s)

并行的理由：

* 两条分支**可以并行执行**（串行的关键路径更长）；
* 通道预算可以直接分配（Attention 与 SSM 的头数独立可调）；
* 一层同时获得"显式 KV 检索"和"固定状态压缩"两种能力。

代价：两条分支都要读输入；kernel 融合更复杂；Attention 的 KV Cache 依然存在
（所以它**不是**"消除 KV Cache"的方案，而是"按比例削减"）。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import Attention

__all__ = ["ParallelHybridMixer"]


class ParallelHybridMixer(nn.Module):
    """Attention 分支 + SSM 分支并行，按通道拼接。

    :param attn_ratio: Attention 分支占的通道比例（其余给 SSM），默认 0.5
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0,
                 attn_ratio: float = 0.5, ssm_kind: str = "mamba2",
                 ssm_state_size: int = 32) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.attn_ratio = float(attn_ratio)
        self.ssm_kind = ssm_kind

        n_attn = max(int(round(cfg.n_heads * self.attn_ratio)), 1)
        n_ssm = max(cfg.n_heads - n_attn, 1)
        self.n_attn_heads, self.n_ssm_heads = n_attn, n_ssm

        # ---- Attention 分支用自己的子配置（头数 = n_attn） ----
        from dataclasses import replace

        a_cfg = replace(cfg, n_heads=n_attn, n_kv_heads=max(min(cfg.n_kv_heads, n_attn), 1))
        self.attn_branch = Attention(a_cfg, layer_idx)

        s_cfg = replace(cfg, ssm_state_size=ssm_state_size)
        if ssm_kind == "mamba2":
            from .sequence_mixer import Mamba2Mixer
            self.ssm_branch = Mamba2Mixer(s_cfg, layer_idx)
        else:
            from .sequence_mixer import GatedDeltaNet
            self.ssm_branch = GatedDeltaNet(s_cfg, layer_idx)

        # 两个分支的输出**都投回 cfg.dim**（各自内部的 o_proj 已经做过），
        # 所以这里是 [dim, dim] → dim 的通道融合，而不是按头数算宽度。
        self.attn_out_dim = cfg.dim
        self.ssm_out_dim = cfg.dim
        self.out_proj = nn.Linear(self.attn_out_dim + self.ssm_out_dim, cfg.dim, bias=False)

    # ------------------------------------------------------------------ #
    @property
    def state_shape(self):
        return getattr(self.ssm_branch, "state_shape", None)

    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="full") -> torch.Tensor:
        a = self.attn_branch(x, positions=positions, rope=rope, cache=cache,
                             attn_mask=attn_mask, is_causal=is_causal,
                             layer_type=layer_type)
        s = self.ssm_branch(x, positions=positions, rope=rope, cache=cache,
                            attn_mask=attn_mask, is_causal=is_causal,
                            layer_type=layer_type)
        return self.out_proj(torch.cat([a, s], dim=-1))

    def extra_repr(self) -> str:
        return (f"attn_heads={self.n_attn_heads}, ssm={self.ssm_kind}, "
                f"ssm_heads={self.n_ssm_heads}, ratio={self.attn_ratio}")
