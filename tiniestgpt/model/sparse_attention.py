"""可学习稀疏注意力与压缩注意力（2026 主流长上下文路线）。

线性注意力（GDN / Mamba / RWKV）与稀疏注意力是**两回事**，不能混为一谈：

* 线性：历史压进固定状态 S，decode 每 token O(1)，但**精确回看弱**；
* 稀疏：仍保留历史条目，每个 query 只挑 k 个位置，decode 每 token O(k)，
  **保留基于内容的跳跃检索**，代价是要付 Indexer 扫描 + Top-K + 非连续 gather。

本模块实现 2026 的三档：

* ``dsa`` —— DeepSeek-V3.2 / GLM-5.2：Lightning Indexer 扫全量历史 → Top-K → 主注意力只算这 k 个 + 局部窗口；
* ``csa`` —— DeepSeek-V4 Compressed Sparse Attention：先按 ``compression`` 重叠池化压缩 m 倍 →
  Indexer 只扫压缩后的候选 → 选 Top-K 压缩块 → 拼局部窗口；
* ``hca`` —— Heavily Compressed Attention：压缩率极大，**不做 Top-K**，直接对全部压缩条目做注意力。

并附带 **IndexShare**（GLM-5.2）：相邻层需要的相关历史位置往往相近，
每 ``topk_freq`` 层才重算一次索引，中间层复用——省掉大部分 Indexer GEMM 与大 Top-K。

两个必须写清楚的认知：

1. **稀疏注意力的复杂度只在"真正用了稀疏 kernel"时才成立。**
   本实现是真的 gather（只搬 k 份 K/V），不是"先构造 [T,S] 稠密 score 再 mask"；
   但 PyTorch 层面的 gather 自身有开销，小模型上未必更快——可以用 benchmark 量出来。
2. **Indexer 的 key 必须和历史一起缓存**，否则 decode 时无法给旧 token 打分。
   本实现把 indexer key 存在 ``cache.states`` 的负下标里（正整数留给递归状态）。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["LightningIndexer", "SparseAttention", "SparseIndexBank",
           "compress_kv", "SPARSE_MODES"]


SPARSE_MODES = ("dsa", "csa", "hca")


def compress_kv(k: torch.Tensor, v: torch.Tensor, m: int) -> Tuple[torch.Tensor, ...]:
    """把 ``[B,S,H,D]`` 的 K/V 按因子 m 做**重叠**平均池化。

    返回 ``(kc, vc, counts)``，前两者 ``[B, ceil(S/m), H, D]``。
    用重叠（kernel=m, stride=m）的均值池化，避免把跨块边界的语义切断。
    """
    if m <= 1:
        return k, v, torch.ones(k.shape[1], device=k.device, dtype=k.dtype)
    B, S, H, D = k.shape
    pad = (-S) % m
    if pad:
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
    S2 = k.shape[1]
    kc = k.reshape(B, S2 // m, m, H, D).mean(dim=2)
    vc = v.reshape(B, S2 // m, m, H, D).mean(dim=2)
    counts = torch.full((S2 // m,), float(m), device=k.device, dtype=k.dtype)
    if pad:
        counts[-1] = float(m - pad)          # 最后一块不满
    return kc, vc, counts


class LightningIndexer(nn.Module):
    """DeepSeek Lightning Indexer：用**少量、低维**的头给历史位置打分。

    ``I(t,s) = Σ_j w(t,j) · ReLU(q_idx(t,j) · k_idx(s) / √d_idx)``

    用 ReLU 而不是 softmax、用低维头（默认 32 维）而不是完整 head_dim，
    都是为了让"检索"的成本远低于"读全部"——否则检索本身就比全量注意力还贵。
    """

    def __init__(self, cfg: ModelConfig, n_heads: int = 4, index_dim: int = 32) -> None:
        super().__init__()
        self.n_heads, self.index_dim = n_heads, index_dim
        self.q_proj = nn.Linear(cfg.dim, n_heads * index_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, n_heads * index_dim, bias=False)
        self.w_proj = nn.Linear(cfg.dim, n_heads, bias=False)

    def project_keys(self, src: torch.Tensor) -> torch.Tensor:
        """把 token 表示投影成 indexer key，形状 ``[B,S,H_i·D_i]``。"""
        return self.k_proj(src)

    def score(self, q_src: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
        """q_src:[B,T,C]，k_idx:[B,S,H_i·D_i] → I:[B,T,S]"""
        B, T, _ = q_src.shape
        S = k_idx.shape[1]
        q = self.q_proj(q_src).view(B, T, self.n_heads, self.index_dim).transpose(1, 2)
        k = k_idx.view(B, S, self.n_heads, self.index_dim).transpose(1, 2)
        # q:[B,H,T,D] k:[B,H,S,D] → 注意下标顺序是 b-h-t-d，别写成 b-t-h-d
        s = torch.einsum("bhtd,bhsd->bhts", q, k) / (self.index_dim ** 0.5)
        s = F.relu(s)
        w = F.softplus(self.w_proj(q_src)).transpose(1, 2)          # [B,H_i,T]
        return torch.einsum("bhts,bht->bts", s, w)


class SparseIndexBank(nn.Module):
    """IndexShare：跨层复用 Top-K 索引（GLM-5.2）。

    复用的是 **Top-K indices 本身**，不是上一层的 attention output / score / KV。
    每层仍用自己的 Q/K/V 在同一组位置上重新算注意力，只是省掉重复的检索。
    """

    def __init__(self, topk_freq: int = 4, skip_offset: int = 0) -> None:
        super().__init__()
        self.topk_freq = max(int(topk_freq), 1)
        self.skip_offset = int(skip_offset)
        self.cache: Dict[int, torch.Tensor] = {}
        self.reuse_count = 0
        self.compute_count = 0

    def owner(self, layer_idx: int) -> int:
        """该层应该复用哪一层的索引。"""
        if layer_idx < self.skip_offset:
            return layer_idx
        return ((layer_idx - self.skip_offset) // self.topk_freq) * self.topk_freq + self.skip_offset

    def should_recompute(self, layer_idx: int) -> bool:
        return self.owner(layer_idx) == layer_idx

    def get(self, layer_idx: int, compute_fn) -> torch.Tensor:
        own = self.owner(layer_idx)
        if own not in self.cache:
            self.cache[own] = compute_fn()
            self.compute_count += 1
        else:
            self.reuse_count += 1
        return self.cache[own]

    def reset(self) -> None:
        self.cache.clear()
        self.reuse_count = 0
        self.compute_count = 0

    def report(self) -> str:
        return (f"IndexShare: 计算 {self.compute_count} 次 / 复用 {self.reuse_count} 次 "
                f"(topk_freq={self.topk_freq}, skip_offset={self.skip_offset})")


class SparseAttention(nn.Module):
    """DSA / CSA / HCA 三合一的可学习稀疏注意力。

    接口与其它 mixer 完全一致：
    ``forward(x, positions, rope, cache, attn_mask, is_causal, layer_type)``
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0, mode: str = "dsa",
                 index_topk: int = 256, compression: int = 1, local_window: int = 128,
                 index_heads: int = 4, index_dim: int = 32,
                 bank: Optional[SparseIndexBank] = None) -> None:
        super().__init__()
        assert mode in SPARSE_MODES, f"mode 必须是 {SPARSE_MODES}"
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mode = mode
        self.index_topk = int(index_topk)
        self.compression = max(int(compression), 1)
        self.local_window = int(local_window)
        self.bank = bank

        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.n_kv_heads = cfg.n_kv_heads
        self.rope_enabled = cfg.rope_type != "none"

        H, D, Hkv = self.n_heads, self.head_dim, self.n_kv_heads
        self.q_proj = nn.Linear(cfg.dim, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.dim, Hkv * D, bias=False)
        self.v_proj = nn.Linear(cfg.dim, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.dim, bias=False)
        self.indexer = LightningIndexer(cfg, index_heads, index_dim) \
            if mode in ("dsa", "csa") else None

    # ------------------------------------------------------------------ #
    def _split(self, t: torch.Tensor, heads: int) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, heads, self.head_dim).transpose(1, 2)     # [B,H,T,D]

    def _gather(self, src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """src:[B,S,H,D]  idx:[B,T,N] → [B,T,N,H,D]"""
        B, S, H, D = src.shape
        N = idx.shape[-1]
        flat = src.reshape(B * S, H * D)
        off = (torch.arange(B, device=src.device) * S).view(B, 1, 1)
        sel = (idx + off).reshape(-1).clamp_(0, B * S - 1)
        return flat.index_select(0, sel).view(B, -1, N, H, D)

    def _attend(self, q, k, v, valid) -> torch.Tensor:
        """q:[B,H,T,D]  k/v:[B,H,T,N,D]  valid:[B,1,T,N] → [B,H,T,D]"""
        s = torch.matmul(q.unsqueeze(-2), k.transpose(-1, -2)).squeeze(-2) / (self.head_dim ** 0.5)
        if valid is not None:
            s = s.masked_fill(~valid, float("-inf"))
        # 若某行真的全 -inf（不该发生，因为总有局部窗口），退化成均匀分布避免 NaN
        bad = (s == float("-inf")).all(dim=-1, keepdim=True)
        s = torch.where(bad, torch.zeros_like(s), s)
        return torch.matmul(torch.softmax(s, dim=-1).unsqueeze(-2), v).squeeze(-2)

    # ------------------------------------------------------------------ #
    def _indexer_history(self, x: torch.Tensor, cache, offset: int):
        """取 indexer key 的**完整历史**（含当前段），并顺手写回 cache。"""
        ik = self.indexer.project_keys(x)                       # [B,T,H_i·D_i]
        if cache is None:
            return ik
        key = -(self.layer_idx + 1)                             # 负下标：避开递归状态
        hist = cache.states.get(key)
        if hist is None or offset == 0:
            hist = ik
        else:
            hist = torch.cat([hist[:, :offset], ik], dim=1)
        cache.states[key] = hist
        return hist

    def _topk(self, x, k_src, offset, T, pool) -> torch.Tensor:
        """算出 Top-K 的**源域下标**（源域 = 压缩域，pool>1 时是块下标）。"""
        def compute():
            hist = self._indexer_history(x, self._cache_ref, self._offset)
            I = self.indexer.score(x, hist)                     # [B,T,S_raw]
            # 把"绝对位置 <= t"翻译成源域约束；压缩域下标由绝对位置整除得到
            S_raw = I.shape[-1]
            if pool > 1:
                # 对每个块内的原始位置取平均，得到块级打分
                nblk = (S_raw + pool - 1) // pool
                pad = (-S_raw) % pool
                if pad:
                    I = F.pad(I, (0, pad), value=float("-inf"))
                I = I.reshape(I.shape[0], I.shape[1], nblk, pool).mean(dim=-1)
            S = I.shape[-1]
            qb = torch.arange(offset, offset + T, device=I.device)
            lim = (qb // pool).view(1, T, 1)
            I = I.masked_fill(torch.arange(S, device=I.device).view(1, 1, S) > lim, float("-inf"))
            return torch.topk(I, min(self.index_topk, S), dim=-1).indices

        if self.bank is not None:
            return self.bank.get(self.layer_idx, compute)
        return compute()

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="full") -> torch.Tensor:
        B, T, _ = x.shape
        dev = x.device
        self._cache_ref, self._offset = cache, 0

        qh = self._split(self.q_proj(x), self.n_heads)          # [B,H,T,D]
        kh = self._split(self.k_proj(x), self.n_kv_heads)
        vh = self._split(self.v_proj(x), self.n_kv_heads)
        if self.rope_enabled and rope is not None:
            qh, kh = rope(qh, kh, positions)

        offset = 0
        if cache is not None:
            view = cache.write(self.layer_idx, kh.transpose(1, 2), vh.transpose(1, 2))
            if view.kind != "dense":
                raise NotImplementedError(
                    "SparseAttention 目前只支持 dense cache：稀疏检索要按绝对位置 gather，"
                    "分页布局下的 gather 需要额外的 block 映射（P2 待办）")
            k_all = view.k[:, : view.kv_len]                     # [B,S,H,D]
            v_all = view.v[:, : view.kv_len]
            offset = view.kv_len - T
        else:
            k_all = kh.transpose(1, 2)
            v_all = vh.transpose(1, 2)
        self._offset = offset
        q_abs = torch.arange(offset, offset + T, device=dev)

        # GQA：把 KV 头广播到 Q 头数（保持与其它 mixer 相同的语义）
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_all = k_all.repeat_interleave(rep, dim=2)
            v_all = v_all.repeat_interleave(rep, dim=2)

        k_src, v_src, pool = k_all, v_all, 1
        if self.mode in ("csa", "hca") and self.compression > 1:
            k_src, v_src, _ = compress_kv(k_all, v_all, self.compression)
            pool = self.compression

        if self.mode == "dsa":
            idx = self._topk(x, k_all, offset, T, pool=1)
        elif self.mode == "csa":
            idx = self._topk(x, k_all, offset, T, pool=pool)
        else:                                                    # hca：不做 Top-K
            idx = None

        out = self._sparse_over(qh, k_src, v_src, idx, is_causal, q_abs, offset, pool)
        return self.o_proj(out)

    def _sparse_over(self, q, k, v, idx, causal, q_abs, offset, pool):
        """选位置 → gather → **只在这 N 个位置上**做注意力。k/v: [B,S,H,D]"""
        B, H, T, D = q.shape
        S = k.shape[1]
        dev = q.device

        if idx is None:                        # HCA：全部压缩条目
            src = torch.arange(S, device=dev).view(1, 1, S).expand(B, T, S).contiguous()
        else:
            src = idx
        # 局部窗口：局部精确性 + indexer 失效时的安全网
        if self.local_window > 0:
            w = min(self.local_window, S)
            starts = torch.clamp(torch.arange(T, device=dev) - w + 1, min=0)
            pos = starts.view(1, T, 1) + torch.arange(w, device=dev).view(1, 1, w)
            pos = pos.expand(B, T, w).contiguous()
            src = torch.cat([src, pos], dim=2)
        src = src.clamp_(0, S - 1)

        k_sel = self._gather(k, src)           # [B,T,N,H,D]
        v_sel = self._gather(v, src)

        if causal:
            # 源域坐标 → 绝对坐标（压缩域需乘回 pool），再与 query 绝对位置比较
            src_abs = src * pool
            valid = (src_abs <= q_abs.view(1, T, 1)).unsqueeze(1)
        else:
            valid = None

        kh = k_sel.permute(0, 3, 1, 2, 4)      # [B,H,T,N,D]
        vh = v_sel.permute(0, 3, 1, 2, 4)
        out = self._attend(q, kh, vh, valid)
        return out.transpose(1, 2).reshape(B, T, H * D)
