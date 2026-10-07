"""纯 PyTorch 的**参考实现**（reference kernels）。

它们存在的意义：
  1. 作为 Triton / CUDA 内核的**正确性基准**（tests 里做数值比对）；
  2. 在没有 GPU 或没有 Triton 的环境里保证系统仍可运行；
  3. 可读性优先——先看懂这里，再看 ``inference/kernels/`` 里的高性能版本。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "build_attention_mask", "sdpa_attention", "naive_attention",
    "paged_attention_ref", "paged_attention_quantized_ref", "repeat_kv",
]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA/MQA：把 KV 头复制 n_rep 份以匹配 Q 头数。x: [B, Hkv, T, D]"""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


def build_attention_mask(q_len: int, kv_len: int, offset: int = 0, window: int = -1,
                         sinks: int = 0, device=None, dtype=torch.bool,
                         bidirectional: bool = False) -> torch.Tensor:
    """构造 [q_len, kv_len] 的可见性掩码。

    :param offset: 第 0 个 query 在全局序列中的下标（prefill=0，decode=kv_len-1）
    :param window: 滑动窗口大小；-1 表示不限制
    :param sinks : 永远可见的前 k 个 key（Attention Sink）
    """
    if window <= 0 and sinks <= 0 and not bidirectional:
        # 纯因果：交给 SDPA 的 is_causal，零额外显存
        return torch.ones(1, device=device, dtype=dtype)  # 占位（调用方需判断）
    q_idx = torch.arange(q_len, device=device) + offset      # [Q]
    k_idx = torch.arange(kv_len, device=device)              # [K]
    if bidirectional:
        mask = torch.ones(q_len, kv_len, device=device, dtype=dtype)
        return mask
    causal = k_idx[None, :] <= q_idx[:, None]
    if window > 0:
        in_window = (q_idx[:, None] - k_idx[None, :]) < window
        if sinks > 0:
            in_window = in_window | (k_idx[None, :] < sinks)
        mask = causal & in_window
    elif sinks > 0:
        mask = causal
    else:
        mask = causal
    return mask.to(dtype)


def naive_attention(q, k, v, scale: Optional[float] = None, mask: Optional[torch.Tensor] = None,
                    softcap: float = 0.0, dropout_p: float = 0.0, training: bool = False) -> torch.Tensor:
    """最朴素的 attention：显式 materialize ``T×T`` 的 logits。

    教学价值最高（一眼看懂），但显存 O(T²)、访存 O(T²)，只适合对照。
    q/k/v: [B, H, T, D]
    """
    scale = scale or (1.0 / math.sqrt(q.shape[-1]))
    if q.shape[1] != k.shape[1]:                      # GQA/MQA
        group = q.shape[1] // k.shape[1]
        k = repeat_kv(k, group)
        v = repeat_kv(v, group)
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if softcap > 0:
        logits = torch.tanh(logits / softcap) * softcap
    if mask is not None and mask.numel() > 1:
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)
    if dropout_p > 0 and training:
        probs = F.dropout(probs, p=dropout_p)
    return torch.einsum("bhqk,bhkd->bhqd", probs.to(v.dtype), v)


def sdpa_attention(q, k, v, scale: Optional[float] = None, mask: Optional[torch.Tensor] = None,
                   softcap: float = 0.0, dropout_p: float = 0.0, training: bool = False,
                   is_causal: bool = True, window: int = -1, sinks: int = 0,
                   offset: int = 0, backend: str = "sdpa") -> torch.Tensor:
    """默认的稠密 attention 后端。

    优先走 ``F.scaled_dot_product_attention``（内部会选 Flash / MemEfficient / Math），
    需要 softcap 或自定义掩码时回退到显式实现。
    """
    if softcap > 0 or backend == "naive":
        m = None
        if mask is not None and mask.numel() > 1:
            m = mask
        elif is_causal or window > 0:
            m = build_attention_mask(q.shape[2], k.shape[2], offset=offset,
                                     window=window, sinks=sinks, device=q.device)
        return naive_attention(q, k, v, scale=scale, mask=m, softcap=softcap,
                               dropout_p=dropout_p, training=training)

    attn_mask = None
    if mask is not None and mask.numel() > 1 and mask.dim() >= 2:
        attn_mask = mask
    elif window > 0 or sinks > 0:
        attn_mask = build_attention_mask(q.shape[2], k.shape[2], offset=offset,
                                         window=window, sinks=sinks, device=q.device)
    elif is_causal and offset > 0:
        # 使用 KV Cache 时，因果掩码必须**带上 offset**：
        # SDPA 的 is_causal 只知道 "j <= i"，会把 offset 之前的 key 全部误伤。
        q_idx = torch.arange(q.shape[2], device=q.device) + offset
        k_idx = torch.arange(k.shape[2], device=q.device)
        attn_mask = k_idx[None, :] <= q_idx[:, None]

    # GQA/MQA：Q 头数是 KV 头数的整数倍。
    # 只有在**不需要显式掩码**时才能走 SDPA 原生的 enable_gqa（省一次 expand）；
    # 需要掩码时必须先 repeat_kv，否则掩码会被丢掉（这是个很容易踩的坑）。
    if q.shape[1] != k.shape[1]:
        group = q.shape[1] // k.shape[1]
        if attn_mask is None:
            try:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout_p if training else 0.0,
                    is_causal=is_causal, scale=scale, enable_gqa=True)
            except TypeError:
                pass
        k = repeat_kv(k, group)
        v = repeat_kv(v, group)

    if attn_mask is not None and attn_mask.dtype is torch.bool:
        # SDPA 需要 float mask（True = 参与计算）
        attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
            ~attn_mask, torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal and attn_mask is None, scale=scale,
    )


def paged_attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale: Optional[float] = None,
                        causal: bool = True, window: int = -1, sinks: int = 0,
                        softcap: float = 0.0, max_len: int = -1) -> torch.Tensor:
    """PagedAttention 的参考实现：**批量 gather + 一次 SDPA**。

    逐序列写 Python 循环也能正确，但会成为 decode 的瓶颈。
    这里改成：把 block table 展平后一次性 gather 出 [B, Lmax, Hkv, D]，
    再用一个带掩码的 SDPA 完成全部计算——既保留了分页的显存优势，
    又让"每个序列长度不同"这件事通过 mask 而不是循环来表达。

    :param q:          [B, H, T, D]，T=1 为 decode，T>1 为（分块）prefill
    :param k_cache:    [num_blocks, block_size, Hkv, D]
    :param v_cache:    同 k_cache
    :param block_table:[B, max_blocks_per_seq]，未使用的槽位为 -1
    :param seq_lens:   [B]，每个序列的 KV 总长度（含本次写入的 token）
    """
    B, H, T, D = q.shape
    Hkv = k_cache.shape[2]
    group = H // Hkv
    scale = scale or (1.0 / math.sqrt(D))
    device = q.device

    # max_len > 0 时直接用（CUDA Graph 需要避免 .item() 带来的 host 同步）
    L = max_len if max_len > 0 else (int(seq_lens.max().item()) if seq_lens.numel() else 0)
    if L == 0:
        return torch.zeros_like(q)

    n_slots = block_table.shape[1] * k_cache.shape[1]
    flat = block_table.reshape(B, -1).clamp(min=0)                     # [B, nblk]
    k_all = k_cache[flat].reshape(B, n_slots, Hkv, D)[:, :L]           # [B,L,Hkv,D]
    v_all = v_cache[flat].reshape(B, n_slots, Hkv, D)[:, :L]
    if group > 1:
        k_all = repeat_kv(k_all.transpose(1, 2), group)                # [B,H,L,D]
        v_all = repeat_kv(v_all.transpose(1, 2), group)

    # ---- 掩码：有效长度 × 因果 × (窗口 ∪ sink) ----
    k_idx = torch.arange(L, device=device)
    valid = k_idx[None, None, :] < seq_lens[:, None, None].to(device)  # [B,1,L]
    q_idx = torch.arange(T, device=device)[None, :, None] + (seq_lens[:, None, None].to(device) - T)
    mask = torch.ones(B, T, L, device=device, dtype=torch.bool)
    if causal:
        mask = mask & (k_idx[None, None, :] <= q_idx)
    if window > 0:
        in_w = (q_idx - k_idx[None, None, :]) < window
        if sinks > 0:
            in_w = in_w | (k_idx[None, None, :] < sinks)
        mask = mask & in_w
    mask = mask & valid                                                # [B,T,L]

    if softcap > 0:
        logits = torch.einsum("bhtd,bhld->bhtl", q.float(), k_all.float()) * scale
        logits = torch.tanh(logits / softcap) * softcap
        logits = logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1)
        return torch.einsum("bhtl,bhld->bhtd", probs.to(v_all.dtype), v_all)

    float_mask = torch.zeros(B, 1, T, L, device=device, dtype=q.dtype).masked_fill(
        ~mask.unsqueeze(1), torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=float_mask, scale=scale)


def paged_attention_quantized_ref(q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens,
                                  scale: Optional[float] = None, causal: bool = True,
                                  window: int = -1, sinks: int = 0) -> torch.Tensor:
    """KV Cache 量化版 PagedAttention：gather **int8** 后再反量化。

    只有"被 gather 到的块"被反量化，因此访存量是 int8 的，
    这就是 KV 量化为什么能直接减少 decode 的带宽瓶颈。
    """
    B, H, T, D = q.shape
    Hkv = k_cache.shape[2]
    group = H // Hkv
    device = q.device
    L = int(seq_lens.max().item()) if seq_lens.numel() else 0
    if L == 0:
        return torch.zeros_like(q)

    flat = block_table.reshape(B, -1).clamp(min=0)
    n_slots = flat.shape[1] * k_cache.shape[1]
    kq = k_cache[flat].reshape(B, n_slots, Hkv, D)[:, :L].to(q.dtype)
    vq = v_cache[flat].reshape(B, n_slots, Hkv, D)[:, :L].to(q.dtype)
    ks = k_scale[flat].reshape(B, n_slots, *k_scale.shape[-2:])[:, :L].to(q.dtype)
    vs = v_scale[flat].reshape(B, n_slots, *v_scale.shape[-2:])[:, :L].to(q.dtype)
    k_all = kq * ks
    v_all = vq * vs

    if group > 1:
        k_all = repeat_kv(k_all.transpose(1, 2), group)
        v_all = repeat_kv(v_all.transpose(1, 2), group)

    k_idx = torch.arange(L, device=device)
    q_idx = torch.arange(T, device=device)[None, :, None] + (seq_lens[:, None, None].to(device) - T)
    mask = torch.ones(B, T, L, device=device, dtype=torch.bool)
    if causal:
        mask = mask & (k_idx[None, None, :] <= q_idx)
    if window > 0:
        in_w = (q_idx - k_idx[None, None, :]) < window
        if sinks > 0:
            in_w = in_w | (k_idx[None, None, :] < sinks)
        mask = mask & in_w
    mask = mask & (k_idx[None, None, :] < seq_lens[:, None, None].to(device))

    float_mask = torch.zeros(B, 1, T, L, device=device, dtype=q.dtype).masked_fill(
        ~mask.unsqueeze(1), torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=float_mask, scale=scale)
