"""稀疏 MoE（Mixture-of-Experts）：用**条件计算**把参数量与计算量解耦。

现代 MoE 的三件套（DeepSeek / Qwen3 / Mixtral 的共同做法）：

1. **细粒度专家 + top-k 路由**：每个 token 只走 k 个专家，
   FLOPs 与 k 成正比，参数量却随专家数线性增长。
2. **共享专家**（shared expert）：总有 1~2 个专家被所有 token 使用，
   承载"通用知识"，避免路由专家重复学习基础能力 → 明显更稳。
3. **无辅助损失的负载均衡**（DeepSeek V3）：
   传统做法加 ``aux_loss``（强制均匀），但会伤害模型质量；
   V3 改为给每个专家一个**可动态更新的 bias**，
   "忙的专家降低 bias、闲的专家提高 bias"，不干扰梯度。
   两种都实现了，用 ``moe_aux_coef`` 切换（0 = 用 bias 方案）。

实现要点：路由后按专家 **sort + 分组** 做 batched GEMM，
而不是对每个专家单独启动一次 kernel——这是吞吐的关键。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mlp import FeedForward

__all__ = ["MoEStats", "MoE"]


@dataclass
class MoEStats:
    """一次 forward 的路由统计（用于监控"专家崩塌"）。"""
    load: torch.Tensor                 # [E] 每个专家分配到的 token 数
    mean_prob: torch.Tensor            # [E] 每个专家的平均路由概率
    dropped: int = 0
    aux_loss: Optional[torch.Tensor] = None
    max_violation: float = 0.0         # |load - mean| / mean 的最大值，越小越均衡


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        self.score_func = cfg.moe_score_func
        self.aux_coef = cfg.moe_aux_coef
        self.bias_lr = cfg.moe_bias_lr
        self.capacity_factor = cfg.moe_capacity_factor

        m = cfg.multiple_of
        expert_hidden = max(m, ((cfg.hidden_dim // 2) // m) * m)
        self.expert_hidden = expert_hidden

        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(cfg.dim, expert_hidden, cfg.act_type, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])
        self.shared = nn.ModuleList([
            FeedForward(cfg.dim, expert_hidden, cfg.act_type, cfg.dropout)
            for _ in range(cfg.n_shared_experts)
        ])
        # 无辅助损失方案所用的动态 bias（不参与梯度）
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("expert_load_acc", torch.zeros(cfg.n_experts))

    # ------------------------------------------------------------------ #
    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (topk_idx [N,k], weights [N,k], probs [N,E])。"""
        logits = self.gate(x).float()                      # [N, E]
        if self.score_func == "sigmoid":
            probs = torch.sigmoid(logits)
            sel = probs + self.expert_bias
        else:                                              # softmax（Switch/Mixtral）
            probs = torch.softmax(logits, dim=-1)
            sel = probs + self.expert_bias
        topk_vals, topk_idx = torch.topk(sel, self.top_k, dim=-1)
        weights = probs.gather(1, topk_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_idx, weights, probs

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_bias(self, load: torch.Tensor) -> None:
        """DeepSeek V3 式均衡：忙的降 bias，闲的升 bias。"""
        mean = load.float().mean()
        err = load.float() - mean
        self.expert_bias -= torch.sign(err) * self.bias_lr
        self.expert_bias -= 0.0 * self.expert_bias           # 保持形状
        self.expert_bias.clamp_(-1.0, 1.0)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, MoEStats]:
        """x: [B, T, C] → (out, aux_loss, stats)"""
        B, T, C = x.shape
        N = B * T
        xf = x.reshape(N, C)

        topk_idx, weights, probs = self._route(xf)           # [N,k]
        E = self.n_experts

        flat_idx = topk_idx.reshape(-1)                      # [N*k]
        flat_w = weights.reshape(-1)
        flat_x = xf.unsqueeze(1).expand(N, self.top_k, C).reshape(N * self.top_k, C)
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        # ---------- 容量限制（可选）：每个专家最多接收 capacity 个 token ----------
        dropped = 0
        if self.capacity_factor > 0:
            capacity = max(int(self.capacity_factor * N * self.top_k / E), self.top_k)
            tmp_order = torch.argsort(flat_idx, stable=True)
            cnts = torch.bincount(flat_idx, minlength=E)
            starts = torch.cumsum(cnts, 0) - cnts                       # 每个专家的起始下标
            rank = torch.arange(flat_idx.numel(), device=x.device) - starts[flat_idx[tmp_order]]
            pos = torch.empty_like(flat_idx)
            pos[tmp_order] = rank                                        # 该 token 在专家内的序号
            keep = pos < capacity
            dropped = int((~keep).sum())
            flat_idx, flat_w, flat_x, token_ids = (
                flat_idx[keep], flat_w[keep], flat_x[keep], token_ids[keep])

        # ---------- 按专家分组（只 sort 一次，之后连续切片 = 分组 GEMM） ----------
        order = torch.argsort(flat_idx)
        sorted_x = flat_x[order]
        sorted_idx = flat_idx[order]
        sorted_w = flat_w[order]
        sorted_token = token_ids[order]
        counts = torch.bincount(sorted_idx, minlength=E).tolist()

        out_sorted = torch.empty_like(sorted_x)
        offset = 0
        for e, cnt in enumerate(counts):
            if cnt == 0:
                continue
            chunk = sorted_x[offset:offset + cnt]
            out_sorted[offset:offset + cnt] = self.experts[e](chunk)
            offset += cnt

        out_flat = out_sorted * sorted_w.unsqueeze(-1)
        agg = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        agg.index_add_(0, sorted_token, out_flat)                        # 按 token 聚合 top-k

        # ---------- 共享专家 ----------
        shared_out = None
        for s in self.shared:
            o = s(xf)
            shared_out = o if shared_out is None else shared_out + o
        if shared_out is not None:
            agg = agg + shared_out

        # ---------- 统计与均衡 ----------
        with torch.no_grad():
            load = torch.bincount(flat_idx, minlength=E).float()
            mean_prob = probs.mean(0)
            if self.training and self.aux_coef <= 0:
                self._update_bias(load)
            self.expert_load_acc = 0.99 * self.expert_load_acc + 0.01 * load

        aux = x.new_zeros(())
        if self.aux_coef > 0:
            # Switch Transformer 的辅助损失：E · Σ f_i · P_i
            f = load / max(load.sum(), 1)
            aux = self.aux_coef * E * (f * mean_prob).sum()

        max_v = float(((load - load.mean()).abs() / max(load.mean(), 1e-9)).max())
        stats = MoEStats(load=load.detach(), mean_prob=mean_prob.detach(),
                         dropped=dropped, aux_loss=aux.detach(), max_violation=max_v)
        return agg.view(B, T, C), aux, stats
