"""Triton 内核：PagedAttention(decode) / 融合 RMSNorm / 融合 SwiGLU。

**为什么 decode 要专门写内核**：
decode 阶段每个序列只有 1 个 query，但要读 L 个 key/value——
这是一个典型的 **memory-bound + 在 (seq, head) 上高度并行** 的算子。
PyTorch 版本为了通用性会 materialize 出 [B, 1, H, L] 的 logits 等临时张量，
而专用内核用 **online softmax**（FlashAttention 的核心技巧）在寄存器里维护
running max / running sum，**完全不写回中间结果**，中间显存从 O(L) 降到 O(1)。
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:  # pragma: no cover - 平台相关
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False

__all__ = ["TRITON_AVAILABLE", "register_triton_kernels", "available_kernels",
           "paged_attention_triton", "rms_norm_triton", "swiglu_triton"]


# --------------------------------------------------------------------------- #
# PagedAttention (decode, T=1)
# --------------------------------------------------------------------------- #
if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attn_decode_kernel(
        Q, K, V, Out, BlockTable, SeqLens,
        stride_qb, stride_qh, sm_scale,
        H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BT_STRIDE: tl.constexpr,
        WINDOW: tl.constexpr, SINKS: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head = head_idx // GROUP
        seq_len = tl.load(SeqLens + seq_idx).to(tl.int32)

        offs_d = tl.arange(0, D)
        q = tl.load(Q + seq_idx * stride_qb + head_idx * stride_qh + offs_d).to(tl.float32)

        m_i = -1.0e30
        l_i = 0.0
        acc = tl.zeros([D], dtype=tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < seq_len
            # block_table[seq, offs_n // BLOCK_N]
            blk = tl.load(BlockTable + seq_idx * BT_STRIDE + offs_n // BLOCK_N,
                          mask=valid, other=0)
            slot = blk * (BLOCK_N * HKV * D) + (offs_n % BLOCK_N) * (HKV * D) \
                + kv_head * D + offs_d
            k = tl.load(K + slot, mask=valid[:, None], other=0.0)
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale

            gi = seq_len - 1
            if WINDOW > 0:
                ok = (gi - offs_n) < WINDOW
                if SINKS > 0:
                    ok = ok | (offs_n < SINKS)
                qk = tl.where(ok & valid, qk, -1.0e30)
            else:
                qk = tl.where(valid, qk, -1.0e30)

            m_new = tl.maximum(m_i, tl.max(qk, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            acc = acc * alpha
            v = tl.load(V + slot, mask=valid[:, None], other=0.0)
            acc += tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        acc = acc / tl.maximum(l_i, 1.0e-9)
        tl.store(Out + seq_idx * stride_qb + head_idx * stride_qh + offs_d,
                 acc.to(Out.dtype.element_ty))

    @triton.jit
    def _rms_norm_kernel(X, W, Y, stride, N: tl.constexpr, eps: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        x = tl.load(X + row * stride + cols).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x, 0) / N + eps)
        w = tl.load(W + cols).to(tl.float32)
        tl.store(Y + row * stride + cols, (x / rms * w).to(Y.dtype.element_ty))

    @triton.jit
    def _swiglu_kernel(A, C, stride, N: tl.constexpr):
        """A = [..., 2N] → silu(gate) * up"""
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        g = tl.load(A + row * stride + cols).to(tl.float32)
        u = tl.load(A + row * stride + N + cols).to(tl.float32)
        y = (g * tl.sigmoid(g)) * u
        tl.store(C + row * stride + cols, y.to(C.dtype.element_ty))


# --------------------------------------------------------------------------- #
def _ref_paged(q, k_cache, v_cache, block_table, seq_lens, scale, causal, window, sinks,
               softcap, max_len=-1):
    from ...model.kernels import paged_attention_ref

    return paged_attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale,
                               causal, window, sinks, softcap, max_len)


def paged_attention_triton(q, k_cache, v_cache, block_table, seq_lens, scale=None,
                           causal=True, window=-1, sinks=0, softcap=0.0, max_len: int = -1):
    """Triton 版 PagedAttention。仅 decode(T=1)+CUDA+无 softcap 时走内核，其余回退。"""
    B, H, T, D = q.shape
    if (not TRITON_AVAILABLE or q.device.type != "cuda" or T != 1 or softcap > 0
            or k_cache.dtype != v_cache.dtype):
        return _ref_paged(q, k_cache, v_cache, block_table, seq_lens, scale,
                          causal, window, sinks, softcap, max_len)

    q = q.contiguous()
    out = torch.empty_like(q)
    scale = scale or (1.0 / math.sqrt(D))
    block_size = k_cache.shape[1]
    HKV = k_cache.shape[2]
    if D not in (16, 32, 64, 128, 256) or block_size not in (8, 16, 32, 64):
        return _ref_paged(q, k_cache, v_cache, block_table, seq_lens, scale,
                          causal, window, sinks, softcap, max_len)

    _paged_attn_decode_kernel[(B, H)](
        q, k_cache, v_cache, out, block_table, seq_lens,
        q.stride(0), q.stride(1), float(scale),
        H=H, HKV=HKV, D=D, GROUP=H // HKV, BLOCK_N=block_size,
        BT_STRIDE=block_table.shape[1],
        WINDOW=int(window), SINKS=int(sinks),
        num_warps=4, num_stages=2,
    )
    return out


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not TRITON_AVAILABLE or x.device.type != "cuda":
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return ((xf * rms) * weight.float()).to(dtype)
    N = x.shape[-1]
    out = torch.empty_like(x)
    _rms_norm_kernel[(x.numel() // N,)](x, weight, out, N, N=N, eps=eps, num_warps=4)
    return out


def swiglu_triton(a: torch.Tensor) -> torch.Tensor:
    """a: [..., 2N] → silu(gate) * up"""
    if not TRITON_AVAILABLE or a.device.type != "cuda":
        import torch.nn.functional as F

        g, u = a.chunk(2, dim=-1)
        return F.silu(g) * u
    N = a.shape[-1] // 2
    out = torch.empty(a.shape[:-1] + (N,), dtype=a.dtype, device=a.device)
    _swiglu_kernel[(a.numel() // a.shape[-1],)](a, out, a.shape[-1], N=N, num_warps=4)
    return out


def register_triton_kernels() -> None:
    """把 Triton 实现注册到内核分发表（由 ``model/dispatch`` 延迟调用）。

    注意：FlashAttention 只是**登记**成可选后端，默认不会被 auto 选中
    （auto 的顺序是 sdpa → flash），因为它走 fp32/ieee 精度、没有反向，
    只有显式指定 ``attn_backend="flash"`` 时才启用。
    """
    if not TRITON_AVAILABLE:
        return
    from ...model.dispatch import register_paged_attention
    from .flash_attention import register_flash_attention

    register_paged_attention("triton", paged_attention_triton)
    register_flash_attention()


def available_kernels() -> dict:
    return {
        "triton": TRITON_AVAILABLE,
        "paged_attention_decode": TRITON_AVAILABLE,
        "rms_norm": TRITON_AVAILABLE,
        "swiglu": TRITON_AVAILABLE,
    }
