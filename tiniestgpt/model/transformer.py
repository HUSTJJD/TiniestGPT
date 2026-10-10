"""Transformer 组装：把所有"先进模块"拼成一个可训练、可推理的模型。

结构（每层）::

    x ──► norm1 ──► Mixer(Attention / MLA / Linear) ──► post_norm1 ──┐
    │                                                                ▼
    └─────────────────────────────────────────────────────────────► (+) ──► norm2 ──► FFN/MoE ──► post_norm2 ──► (+)

* ``pre_norm + post_norm``（Sandwich）是 Qwen3 / Gemma2 的做法，
  牺牲一点速度换取深层数值稳定；
* Mixer 类型可按层配置（full / window / linear），实现"混合架构"；
* FFN 可整体替换为稀疏 MoE；
* 所有 KV Cache 交互都通过 ``cache`` 参数注入，模型本身不持有状态
  → 同一份权重既能跑训练，也能跑 PagedAttention 推理引擎。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import Attention
from .config import ModelConfig
from .linear_attention import LinearAttention
from .mla import MultiHeadLatentAttention
from .mlp import FeedForward
from .hyper_connections import HyperConnections, MHCBlock
from .moe import MoE
from .mtp import MTPModule
from .norms import build_norm
from .rope import RotaryEmbedding
from .parallel_mixer import ParallelHybridMixer
from .rwkv7 import RWKV7Mixer
from .sequence_mixer import GatedDeltaNet, Mamba2Mixer
from .sparse_attention import SparseAttention, SparseIndexBank

__all__ = ["TransformerBlock", "Transformer"]


# Transformer.__init__ 构造 layers 时临时放入共享的 IndexShare bank。
# 用模块级槽位传递是为了不改动 TransformerBlock 的构造签名（它要保持简单可读）。
_SPARSE_BANK = [None]


def _sparse_bank(cfg):
    if cfg.sparse_index_share <= 1:
        return None
    return _SPARSE_BANK[0]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, layer_type: str = "full") -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.pre_norm = cfg.pre_norm

        if layer_type == "gdn":
            self.mixer: nn.Module = GatedDeltaNet(cfg, layer_idx)
        elif layer_type == "mamba2":
            self.mixer = Mamba2Mixer(cfg, layer_idx)
        elif layer_type == "rwkv7":
            # 无 Attention、无 KV Cache 的矩阵状态（状态大小与上下文长度无关）
            self.mixer = RWKV7Mixer(cfg, layer_idx)
        elif layer_type == "parallel":
            # 同层并行 Attention + SSM（Falcon-H1）
            self.mixer = ParallelHybridMixer(cfg, layer_idx,
                                             attn_ratio=cfg.parallel_attn_ratio,
                                             ssm_kind=cfg.parallel_ssm)
        elif layer_type == "sparse":
            # DSA / CSA / HCA：可学习稀疏 + 压缩注意力（IndexShare 跨层复用）
            self.mixer = SparseAttention(
                cfg, layer_idx, mode=cfg.sparse_mode, index_topk=cfg.sparse_topk,
                compression=cfg.sparse_compression, local_window=cfg.sparse_local_window,
                index_heads=cfg.sparse_index_heads, index_dim=cfg.sparse_index_dim,
                bank=_sparse_bank(cfg))
        elif layer_type == "linear":
            # "linear" 是可配置的槽位：retention(2023) / gdn / mamba2 三选一
            if cfg.seq_mixer == "gdn":
                self.mixer = GatedDeltaNet(cfg, layer_idx)
            elif cfg.seq_mixer == "mamba2":
                self.mixer = Mamba2Mixer(cfg, layer_idx)
            else:
                self.mixer = LinearAttention(cfg, layer_idx)
        elif cfg.attn_type == "mla":
            self.mixer = MultiHeadLatentAttention(cfg, layer_idx)
        else:
            self.mixer = Attention(cfg, layer_idx)

        self.ffn: nn.Module = MoE(cfg, layer_idx) if cfg.moe_enabled else FeedForward(
            cfg.dim, cfg.hidden_dim, cfg.act_type, cfg.dropout)

        self.norm1 = build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps)
        self.norm2 = build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps)
        self.post_norm1 = build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps,
                                     init_ones=not cfg.post_norm_init_zero) if cfg.post_norm else None
        self.post_norm2 = build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps,
                                     init_ones=not cfg.post_norm_init_zero) if cfg.post_norm else None
        self.aux_loss: Optional[torch.Tensor] = None
        self.moe_stats = None

        # mHC：残差从 1 条流扩成 hc_mult 条，混合矩阵用 Sinkhorn 投影成双随机
        self.hc = HyperConnections(cfg.dim, mult=max(cfg.hc_mult, 1),
                                   iters=cfg.hc_sinkhorn_iters) if cfg.hc_mult > 1 else None

    def _sub_attn(self, x, positions, rope, cache, attn_mask, is_causal):
        h = self.norm1(x) if self.pre_norm else x
        h = self.mixer(h, positions=positions, rope=rope, cache=cache,
                       attn_mask=attn_mask, is_causal=is_causal, layer_type=self.layer_type)
        if self.post_norm1 is not None:
            h = self.post_norm1(h)
        return h

    def _sub_ffn(self, x):
        h = self.norm2(x) if self.pre_norm else x
        aux = None
        if isinstance(self.ffn, MoE):
            h, aux, self.moe_stats = self.ffn(h)
        else:
            h = self.ffn(h)
        if self.post_norm2 is not None:
            h = self.post_norm2(h)
        self.aux_loss = aux
        return h

    def _mixer_and_ffn(self, x, positions, rope, cache, attn_mask, is_causal):
        if self.hc is None:
            x = x + self._sub_attn(x, positions, rope, cache, attn_mask, is_causal)
            x = x + self._sub_ffn(x)
            return x

        # mHC 路径：X 的形状是 [B, L, m, D]
        X = MHCBlock.expand(x, self.hc.mult)
        X = self.hc.post_mix(X, self._sub_attn(self.hc.pre_mix(X), positions, rope,
                                               cache, attn_mask, is_causal))
        X = self.hc.post_mix(X, self._sub_ffn(self.hc.pre_mix(X)))
        return MHCBlock.collapse(X)

    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cfg.gradient_checkpointing and self.training:
            x = checkpoint(self._mixer_and_ffn, x, positions, rope, cache, attn_mask, is_causal,
                           use_reentrant=False)
        else:
            x = self._mixer_and_ffn(x, positions, rope, cache, attn_mask, is_causal)
        return x, self.aux_loss


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.layer_types = cfg.layer_type_list()

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        # MLA 只对"解耦的那部分维度"做旋转，因此 RoPE 的维度与 head_dim 不同
        rope_head_dim = (min(cfg.mla_rope_dim, cfg.head_dim - 1)
                         if cfg.attn_type == "mla" else cfg.head_dim)
        self.rope = RotaryEmbedding(
            head_dim=rope_head_dim, rope_type=cfg.rope_type, theta=cfg.rope_theta,
            scaling=cfg.rope_scaling, original_max_len=cfg.rope_original_max_len,
            max_seq_len=cfg.max_seq_len, partial=cfg.rope_partial, mscale=cfg.rope_mscale,
        ) if cfg.rope_type != "none" else None

        # IndexShare 的 bank 是**跨层共享**的 nn.Module，必须挂在 Transformer 上
        # 才能进 state_dict；放在 block 里会变成每层一份，失去"共享"的意义。
        self.index_bank = SparseIndexBank(topk_freq=max(cfg.sparse_index_share, 1),
                                          skip_offset=cfg.sparse_index_skip) \
            if "sparse" in self.layer_types else None
        _SPARSE_BANK[0] = self.index_bank
        try:
            self.layers = nn.ModuleList([
                TransformerBlock(cfg, i, self.layer_types[i]) for i in range(cfg.n_layers)
            ])
        finally:
            _SPARSE_BANK[0] = None
        self.norm_f = build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # MTP：n 个轻量预测头，共享 lm_head（推理时同时充当推测解码的草稿器）
        self.mtp = MTPModule(cfg, cfg.mtp_n_predict) if cfg.mtp_enabled else None

        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_mtp_logits: Optional[List[torch.Tensor]] = None
        self.init_weights()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_weights(self) -> None:
        cfg = self.cfg
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=cfg.init_std)
        if cfg.depth_scaled_init:
            # 残差分支的输出投影按 1/sqrt(2L) 缩小，避免深层激活值随层数累积放大
            s = 1.0 / math.sqrt(2 * cfg.n_layers)
            for name, p in self.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w_out.weight"):
                    p.mul_(s)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,                     # [B, T]
        positions: Optional[torch.Tensor] = None,
        cache=None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
        return_mtp: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device

        if positions is None:
            start = getattr(cache, "write_pos", 0) if cache is not None else 0
            positions = torch.arange(start, start + T, device=device).unsqueeze(0).expand(B, T)

        x = self.tok_embeddings(input_ids)
        hidden: List[torch.Tensor] = []
        aux_total = None
        for block in self.layers:
            x, aux = block(x, positions=positions, rope=self.rope, cache=cache,
                           attn_mask=attn_mask, is_causal=is_causal)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux
            if output_hidden_states:
                hidden.append(x)
        self.last_aux_loss = aux_total

        x = self.norm_f(x)
        if logits_to_keep > 0:
            x = x[:, -logits_to_keep:, :]
        logits = self.lm_head(x)
        # MTP：只在需要时算（推理的草稿阶段 / 训练算损失时）
        self.last_mtp_logits = self.mtp(x, self.lm_head) if (
            self.mtp is not None and (return_mtp or self.training)) else None
        if output_hidden_states:
            return logits, hidden
        return logits

    # ------------------------------------------------------------------ #
    # 工程辅助
    # ------------------------------------------------------------------ #
    def num_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed = self.tok_embeddings.weight.numel()
        return {"total": total, "trainable": trainable, "non_embedding": total - embed}

    @property
    def active_params(self) -> int:
        """MoE 场景下每个 token 真正参与计算的参数量。"""
        if not self.cfg.moe_enabled:
            return self.num_params()["total"]
        # 单个专家的参数量（注意不要把所有专家累加起来）
        per_expert = sum(p.numel() for p in self.layers[0].ffn.experts[0].parameters())
        shared = per_expert * self.cfg.n_shared_experts
        others = sum(p.numel() for n, p in self.layers[0].named_parameters()
                     if not n.startswith("ffn."))
        return int(self.cfg.n_layers * (others + self.cfg.n_experts_per_tok * per_expert + shared)
                   + self.tok_embeddings.weight.numel())

    def cache_spec(self) -> Tuple[int, int]:
        """(每层的 KV 头数, 每头的维度)——供推理引擎分配 KV Cache。"""
        if self.cfg.attn_type == "mla":
            # 用真实模块里的（可能被 clamp 过的）维度，保证与实现一致
            mixer = self.layers[0].mixer
            return 1, mixer.latent + mixer.rope_dim
        return self.cfg.n_kv_heads, self.cfg.head_dim

    def kv_cache_bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        h, d = self.cache_spec()
        return int(2 * self.cfg.n_layers * h * d * torch.empty(0, dtype=dtype).element_size())

    def recurrent_state_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        """线性 / SSM 层的递归状态字节数——**按序列计，与上下文长度无关**。

        这是 GDN / Mamba 相对 Full Attention 的核心收益：
        KV Cache 是 ``O(L)``，而这里的固定状态是 ``O(1)``。
        """
        el = torch.empty(0, dtype=dtype).element_size()
        total = 0
        for block in self.layers:
            m = getattr(block, "mixer", None)
            shape = getattr(m, "state_shape", None)
            if shape is None:
                continue
            n = 1
            for s in shape:
                n *= s
            total += n * el
        return int(total)

    @property
    def mixer_histogram(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for block in self.layers:
            m = block.mixer
            name = type(m).__name__
            out[name] = out.get(name, 0) + 1
        return out

    def flops_per_token(self, seq_len: int = 1024) -> float:
        """估算每 token 的前向 FLOPs（用于 MFU）。"""
        n = self.active_params
        non_emb = max(n - self.cfg.vocab_size * self.cfg.dim, 1)
        attn = 4.0 * self.cfg.n_layers * self.cfg.dim * seq_len
        return 2.0 * non_emb + attn

    def summary(self) -> str:
        p = self.num_params()
        lines = [
            f"TiniestGPT  dim={self.cfg.dim} layers={self.cfg.n_layers} "
            f"heads={self.cfg.n_heads}/{self.cfg.n_kv_heads} head_dim={self.cfg.head_dim}",
            f"  params        : {p['total'] / 1e6:.2f} M (non-embedding {p['non_embedding'] / 1e6:.2f} M)",
            f"  active params : {self.active_params / 1e6:.2f} M"
            + (" (MoE)" if self.cfg.moe_enabled else ""),
            f"  vocab / seq   : {self.cfg.vocab_size} / {self.cfg.max_seq_len}",
            f"  norm          : {self.cfg.norm_type} (pre={self.cfg.pre_norm}, post={self.cfg.post_norm}, qk={self.cfg.qk_norm})",
            f"  rope          : {self.cfg.rope_type} (theta={self.cfg.rope_theta}, scale={self.cfg.rope_scaling})",
            f"  attn          : {self.cfg.attn_type} (window={self.cfg.attn_window}, sinks={self.cfg.attn_sinks})",
            f"  ffn           : {'MoE x%d (top-%d)' % (self.cfg.n_experts, self.cfg.n_experts_per_tok)
                                if self.cfg.moe_enabled else self.cfg.act_type}",
            f"  layer types   : {dict((t, self.layer_types.count(t)) for t in set(self.layer_types))}",
            f"  mixers        : {self.mixer_histogram}",
            f"  kv/token      : {self.kv_cache_bytes_per_token() / 1024:.2f} KB @fp16",
            f"  recurrent s/s : {self.recurrent_state_bytes() / 1024:.2f} KB/seq（与上下文长度无关）",
        ]
        return "\n".join(lines)
