"""FlashAttention 的**教学实现**（Triton）。

为什么要自己写一遍：
FlashAttention 的全部魔法就两件事——

1. **Tiling**：外层循环遍历 KV 块、内层把 Q 块留在 SRAM/寄存器里，
   让 QK^T → scale → mask → softmax → PV 全部在片上完成，
   中间结果**一次都不写回 HBM** → HBM 访存从 O(N²) 降到 O(N)；
2. **Online softmax**：边扫边维护 ``(running_max, running_sum)``，
   避免"先求 max 再求 sum"的两趟扫描（见 ``kernels/csrc/04_softmax.cu``）。

与生产版（flash-attn / FlashInfer）的差距，正是本文件刻意保留"可读性"的地方：

* 统一按 **fp32 + ieee 精度** 做 matmul，不走 Tensor Core（生产版必须走）；
* 没有反向传播（需要训练时自动回退到 SDPA）；
* 没有 split-KV / Flash-Decoding（decode 阶段请走 PagedAttention）；
* 不支持 softcap、任意 attn_mask（这些直接回退）。

启用方式（默认仍是 sdpa，避免改变既有基准）：

    ModelConfig(attn_backend="flash")     # 或环境变量 TINIESTGPT_ATTN_BACKEND=flash
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

__all__ = ["TRITON_AVAILABLE", "flash_attention", "register_flash_attention",
           "flash_supported", "flash_attention_blocked_ref"]


def flash_attention_blocked_ref(q, k, v, is_causal: bool = True, window: int = -1,
                                sinks: int = 0, offset: int = 0, block_m: int = 32,
                                block_n: int = 32) -> torch.Tensor:
    """**分块 + online softmax 的纯 PyTorch 版本**（任何设备都能跑）。

    它就是上面 Triton kernel 的逐行翻译：
    外层遍历 Q 块、内层遍历 KV 块，用 ``(m_i, l_i, acc)`` 三个 running 量
    把结果合并出来，全程不 materialize ``T×T`` 的 logits 矩阵。

    存在的意义：Triton 只在 Linux 上可用，而这里是**算法本身**的可执行说明，
    也是数值测试的基准——先确认这个对了，再看 kernel 有没有翻译错。
    """
    B, HQ, Tq, D = q.shape
    HKV = k.shape[1]
    group = HQ // HKV
    Tk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)

    for b in range(B):
        for h in range(HQ):
            kv_h = h // group
            for m0 in range(0, Tq, block_m):
                m1 = min(m0 + block_m, Tq)
                qb = q[b, h, m0:m1].float()                       # [M, D]
                m_i = torch.full((m1 - m0,), float("-inf"), device=q.device)
                l_i = torch.zeros(m1 - m0, device=q.device)
                acc = torch.zeros(m1 - m0, D, device=q.device)
                q_idx = torch.arange(m0, m1, device=q.device) + offset

                hi = min(Tk, m1 + offset) if is_causal else Tk
                for n0 in range(0, hi, block_n):
                    n1 = min(n0 + block_n, Tk)
                    kb = k[b, kv_h, n0:n1].float()                # [N, D]
                    vb = v[b, kv_h, n0:n1].float()
                    qk = (qb @ kb.transpose(0, 1)) * scale        # [M, N]

                    k_idx = torch.arange(n0, n1, device=q.device)
                    ok = torch.ones(1, dtype=torch.bool, device=q.device)
                    if is_causal:
                        ok = ok & (k_idx[None, :] <= q_idx[:, None])
                    if window > 0:
                        in_w = (q_idx[:, None] - k_idx[None, :]) < window
                        if sinks > 0:
                            in_w = in_w | (k_idx[None, :] < sinks)
                        ok = ok & in_w
                    qk = torch.where(ok, qk, torch.tensor(-1e30, device=qk.device))

                    m_new = torch.maximum(m_i, qk.max(dim=1).values)
                    alpha = torch.exp(m_i - m_new)
                    p = torch.exp(qk - m_new[:, None])
                    l_i = l_i * alpha + p.sum(dim=1)
                    acc = acc * alpha[:, None] + p @ vb
                    m_i = m_new

                out[b, h, m0:m1] = (acc / l_i.clamp(min=1e-9)[:, None]).to(out.dtype)
    return out


if TRITON_AVAILABLE:

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O,
        sq_b, sq_h, sq_m, sk_b, sk_h, sk_n, sv_b, sv_h, sv_n, so_b, so_h, so_m,
        Tq, Tk, offset, scale,
        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr, WINDOW: tl.constexpr, SINKS: tl.constexpr,
    ):
        """一个 program 负责一个 (batch·head, Q 块) 的注意力输出。"""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // HQ
        h = pid_bh % HQ
        kv_h = h // (HQ // HKV)                       # GQA / MQA 的头映射

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask_m = offs_m < Tq

        q_ptrs = Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_m + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        q_idx = offs_m + offset                       # query 在整条序列中的全局下标
        m_i = tl.full([BLOCK_M], -1.0e30, tl.float32)   # running max
        l_i = tl.zeros([BLOCK_M], tl.float32)           # running sum
        acc = tl.zeros([BLOCK_M, D], tl.float32)        # running 输出

        # 因果情况下，本 Q 块只需要看到对角线之前的 KV
        if CAUSAL:
            hi = tl.minimum(Tk, (pid_m + 1) * BLOCK_M + offset)
        else:
            hi = Tk

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < Tk

            k_ptrs = K + b * sk_b + kv_h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :]
            v_ptrs = V + b * sv_b + kv_h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :]
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

            ok = mask_n[None, :]
            if CAUSAL:
                ok = ok & (offs_n[None, :] <= q_idx[:, None])
            if WINDOW > 0:
                in_w = (q_idx[:, None] - offs_n[None, :]) < WINDOW
                if SINKS > 0:
                    in_w = in_w | (offs_n[None, :] < SINKS)
                ok = ok & in_w
            qk = tl.where(ok, qk, -1.0e30)

            # ---- online softmax ----
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)                # 把旧的 acc / l 缩放到新基线
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc += tl.dot(p, v, input_precision="ieee")
            m_i = m_new

        acc = acc / tl.maximum(l_i, 1.0e-9)[:, None]
        o_ptrs = O + b * so_b + h * so_h + offs_m[:, None] * so_m + offs_d[None, :]
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def _sdpa(q, k, v, mask, is_causal, window, sinks, offset, softcap,
          dropout_p=0.0, training=False, **kwargs):
    from ...model.kernels import sdpa_attention

    return sdpa_attention(q, k, v, scale=None, mask=mask, softcap=softcap,
                          dropout_p=dropout_p, training=training,
                          is_causal=is_causal, window=window, sinks=sinks, offset=offset)


def flash_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    softcap: float = 0.0, mask: Optional[torch.Tensor] = None) -> bool:
    """当前输入能否走 FlashAttention 内核（否则回退 SDPA）。"""
    if not TRITON_AVAILABLE or q.device.type != "cuda":
        return False
    if softcap and softcap > 0:
        return False
    if mask is not None and torch.is_tensor(mask) and mask.numel() > 1:
        return False                                   # 任意 mask 暂不支持
    HQ, HKV = q.shape[1], k.shape[1]
    if HQ % HKV != 0:
        return False
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return False                                   # 没有反向实现，训练时不能偷偷换掉
    return True


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    mask: Optional[torch.Tensor] = None, is_causal: bool = True,
                    window: int = -1, sinks: int = 0, offset: int = 0,
                    softcap: float = 0.0, dropout_p: float = 0.0,
                    training: bool = False, **kwargs) -> torch.Tensor:
    """FlashAttention（分块 + online softmax）。q/k/v: [B, H, T, D]。"""
    if not flash_supported(q, k, v, softcap, mask) or dropout_p > 0:
        return _sdpa(q, k, v, mask, is_causal, window, sinks, offset, softcap,
                     dropout_p, training)

    B, HQ, Tq, D = q.shape
    HKV = k.shape[1]
    Tk = k.shape[2]
    if Tq == 0 or Tk == 0:
        return torch.empty_like(q)

    def _blk(n: int) -> int:
        # tl.dot 要求最小 16×16；太大则 SRAM 装不下
        return max(16, min(64, triton.next_power_of_2(n)))

    BLOCK_M, BLOCK_N = _blk(Tq), _blk(Tk)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    _flash_fwd_kernel[(triton.cdiv(Tq, BLOCK_M), B * HQ)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        Tq, Tk, int(offset), float(scale),
        HQ=HQ, HKV=HKV, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=bool(is_causal), WINDOW=int(window), SINKS=int(sinks),
        num_warps=4, num_stages=2,
    )
    return out


def register_flash_attention() -> None:
    """把 FlashAttention 注册成稠密注意力的一个后端（名字 ``flash``）。"""
    if not TRITON_AVAILABLE:
        return
    from ...model.dispatch import register_dense_attention

    register_dense_attention("flash", flash_attention)
