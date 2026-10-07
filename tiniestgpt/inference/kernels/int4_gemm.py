"""INT4 权重的**融合 GEMM**：把 dequant 与矩阵乘合并进同一个 kernel。

为什么必须融合（这正是 vLLM 比我们的参考实现快的地方）：

``QuantizedLinear.forward`` 目前是两步——

1. ``dequantize_weight()`` 把 int4 unpack 成 fp16 → **写出一份 [out, in] 的 fp16 权重**；
2. 再用这份 fp16 去做 GEMM → **再读一遍**。

decode 阶段 batch 很小，GEMM 是彻底的 **memory bound**，
于是"多写一次 + 多读一次权重"直接把带宽省下来的收益吃掉一半。
Marlin / Machete 这类 kernel 的做法是：**在寄存器里反量化，权重一次都不落地**。

本文件提供两条路径：

* :func:`fused_int4_gemm`（Triton）：真正的融合 kernel（仅 Linux + CUDA + fp16）
* :func:`fused_int4_gemm_torch`：纯 PyTorch 的"分块反量化"实现，
  至少避免 materialize 一整份 fp16 权重 —— 任何设备都能跑，也是数值基准。

融合 kernel 默认**不启用**（``TINIESTGPT_FUSED_INT4=1`` 或显式传参打开），
因为它在没有实测过的平台上可能编译失败，而参考实现永远正确。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:  # pragma: no cover - 平台相关
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False

__all__ = ["TRITON_AVAILABLE", "fused_int4_gemm", "fused_int4_gemm_torch",
           "int4_gemm_ref", "fused_int4_enabled"]


def fused_int4_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit and TRITON_AVAILABLE
    return bool(os.environ.get("TINIESTGPT_FUSED_INT4")) and TRITON_AVAILABLE


# --------------------------------------------------------------------------- #
# 参考实现：unpack → dequant → mm（与 QuantizedLinear 完全等价）
# --------------------------------------------------------------------------- #
def int4_gemm_ref(packed_w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor,
                  group_size: int, q_shape: Tuple[int, ...],
                  bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from ...inference.quantization.base import unpack_int4

    q = unpack_int4(packed_w, q_shape).to(scales.dtype)              # [out, in]
    g = group_size if group_size > 0 else q.shape[1]
    w = (q.reshape(q.shape[0], q.shape[1] // g, g) * scales.unsqueeze(-1)).reshape(q.shape)
    return torch.nn.functional.linear(x, w.to(x.dtype), bias)


def fused_int4_gemm_torch(packed_w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor,
                          group_size: int, q_shape: Tuple[int, ...],
                          bias: Optional[torch.Tensor] = None,
                          chunk_rows: int = 128) -> torch.Tensor:
    """分块反量化：每次只 dequant ``chunk_rows`` 行权重，立刻用掉。

    相比一次性 ``dequantize_weight()``，峰值显存从 ``out·in·2B`` 降到
    ``chunk_rows·in·2B``，而且 dequant 与 GEMM 在时间上重叠得更好
    —— 这是"不写 kernel 也能拿到的那部分收益"。
    """
    from ...inference.quantization.base import unpack_int4

    out_f, in_f = q_shape
    g = group_size if group_size > 0 else in_f
    x2d = x.reshape(-1, x.shape[-1])
    out = torch.empty(x2d.shape[0], out_f, dtype=x.dtype, device=x.device)

    for r0 in range(0, out_f, chunk_rows):
        r1 = min(r0 + chunk_rows, out_f)
        q = unpack_int4(packed_w[r0:r1], (r1 - r0, in_f)).to(scales.dtype)
        w = (q.reshape(r1 - r0, in_f // g, g)
             * scales[r0:r1].unsqueeze(-1)).reshape(r1 - r0, in_f)
        out[:, r0:r1] = x2d @ w.to(x.dtype).transpose(0, 1)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], out_f)


# --------------------------------------------------------------------------- #
# Triton 融合 kernel：在寄存器里 unpack + 乘 scale，权重不落地
# --------------------------------------------------------------------------- #
if TRITON_AVAILABLE:

    @triton.jit
    def _int4_gemm_kernel(
        X, W, S, O,
        M, N, K,
        stride_xm, stride_wn, stride_sn, stride_om,
        GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """X: [M, K] fp16；W: [N, K//2] int32（每 int32 两个 nibble）；S: [N, K/GROUP]。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        half_k = BLOCK_K // 2

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
        # 打包权重：每个 int32 覆盖 2 个输入通道 → 只需 BLOCK_K/2 个 int32
        w_ptrs = W + offs_n[:, None] * stride_wn + tl.arange(0, half_k)[None, :]
        k_base = tl.arange(0, half_k) * 2                       # 每个 int32 的起始通道

        for k0 in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
                        other=0.0)
            pk = tl.load(w_ptrs, mask=(offs_n[:, None] < N), other=0)

            lo = ((pk & 0xF) - 8).to(tl.float16)
            hi = (((pk >> 4) & 0xF) - 8).to(tl.float16)
            w = tl.interleave(lo, hi)                            # [BLOCK_N, BLOCK_K]

            gidx = (k0 + k_base) // GROUP                        # 每个 int32 所在的组
            sc = tl.load(S + offs_n[:, None] * stride_sn + gidx[None, :],
                         mask=offs_n[:, None] < N, other=0.0)
            # scale 是按"每 2 个通道"取的（同一组内两个通道共享 scale），展开回 BLOCK_K
            sc = tl.interleave(sc, sc)
            w = w * sc

            acc += tl.dot(x, tl.trans(w))
            x_ptrs += BLOCK_K
            w_ptrs += half_k

        tl.store(O + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_int4_gemm(packed_w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor,
                    group_size: int, q_shape: Tuple[int, ...],
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Triton 融合 kernel；任一条件不满足就回退到 :func:`fused_int4_gemm_torch`。"""
    out_f, in_f = q_shape
    ok = (TRITON_AVAILABLE and x.device.type == "cuda" and x.dtype == torch.float16
          and group_size > 0 and in_f % group_size == 0 and in_f % 2 == 0
          and out_f % 32 == 0)
    if not ok:
        return fused_int4_gemm_torch(packed_w, scales, x, group_size, q_shape, bias)

    x2d = x.reshape(-1, in_f).contiguous()
    M = x2d.shape[0]
    out = torch.empty(M, out_f, dtype=torch.float16, device=x.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_f, BLOCK_N))
    _int4_gemm_kernel[grid](
        x2d, packed_w, scales, out, M, out_f, in_f,
        x2d.stride(0), packed_w.stride(0), scales.stride(0), out.stride(0),
        GROUP=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    out = out.reshape(*x.shape[:-1], out_f)
    if bias is not None:
        out = out + bias
    return out
