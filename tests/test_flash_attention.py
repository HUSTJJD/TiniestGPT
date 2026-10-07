"""FlashAttention 的数值一致性测试。

分两层：
  * ``flash_attention_blocked_ref`` 是算法的纯 PyTorch 版，**任何设备都能跑**，
    因此这部分用例永远生效（这是"算法对不对"的验证）；
  * 真正的 Triton kernel 只在 Linux + Triton 可用时才跑（其余情况 skip）。
"""

from __future__ import annotations

import pytest
import torch

from tiniestgpt.inference.kernels.flash_attention import (
    TRITON_AVAILABLE, flash_attention, flash_attention_blocked_ref, flash_supported,
)
from tiniestgpt.model.kernels import naive_attention

requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton 不可用（仅 Linux）")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA 设备")


def _case(b=2, hq=4, hkv=2, tq=37, tk=53, d=32, dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    q = torch.randn(b, hq, tq, d, device=device, dtype=dtype)
    k = torch.randn(b, hkv, tk, d, device=device, dtype=dtype)
    v = torch.randn(b, hkv, tk, d, device=device, dtype=dtype)
    return q, k, v


def _naive(q, k, v, is_causal: bool = True, offset: int = 0,
           window: int = -1, sinks: int = 0) -> torch.Tensor:
    """显式构造可见性掩码后调用 ``naive_attention``（它自己不会默认加因果掩码）。"""
    Tq, Tk = q.shape[2], k.shape[2]
    q_idx = torch.arange(Tq, device=q.device) + offset
    k_idx = torch.arange(Tk, device=q.device)
    mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool)
    if is_causal:
        mask = mask & (k_idx[None, :] <= q_idx[:, None])
    if window > 0:
        in_w = (q_idx[:, None] - k_idx[None, :]) < window
        if sinks > 0:
            in_w = in_w | (k_idx[None, :] < sinks)
        mask = mask & in_w
    return naive_attention(q.float(), k.float(), v.float(), mask=mask)


# --------------------------------------------------------------------------- #
# 算法层（任何设备）
# --------------------------------------------------------------------------- #
def test_blocked_ref_matches_naive_causal():
    q, k, v = _case()
    got = flash_attention_blocked_ref(q, k, v, is_causal=True)
    assert torch.allclose(got.float(), _naive(q, k, v), atol=2e-5, rtol=1e-4)


def test_blocked_ref_matches_naive_non_causal():
    q, k, v = _case(tq=17, tk=17)
    got = flash_attention_blocked_ref(q, k, v, is_causal=False)
    assert torch.allclose(got.float(), _naive(q, k, v, is_causal=False), atol=2e-5, rtol=1e-4)


def test_blocked_ref_gqa_and_mqa():
    for hkv in (2, 1):
        q, k, v = _case(hq=4, hkv=hkv)
        got = flash_attention_blocked_ref(q, k, v, is_causal=True)
        assert torch.allclose(got.float(), _naive(q, k, v), atol=2e-5, rtol=1e-4)


def test_blocked_ref_sliding_window_with_sinks():
    q, k, v = _case(tq=40, tk=40)
    got = flash_attention_blocked_ref(q, k, v, is_causal=True, window=8, sinks=2)
    assert torch.allclose(got.float(), _naive(q, k, v, window=8, sinks=2),
                          atol=2e-5, rtol=1e-4)


def test_blocked_ref_with_offset_matches_full():
    """decode 场景：q 只有 1 个 token，但它在序列中的 offset = kv_len - 1。"""
    q_full, k, v = _case(tq=48, tk=48)
    ref = _naive(q_full, k, v)
    for t in (0, 17, 47):
        q = q_full[:, :, t:t + 1]
        got = flash_attention_blocked_ref(q, k, v, is_causal=True, offset=t)
        assert torch.allclose(got.float(), ref[:, :, t:t + 1], atol=2e-5, rtol=1e-4)


def test_flash_supported_false_on_cpu():
    q, k, v = _case(device="cpu")
    assert not flash_supported(q, k, v)


def test_flash_supported_false_when_grad_needed():
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA 设备")
    q, k, v = _case(device="cuda")
    q.requires_grad_(True)
    assert not flash_supported(q, k, v)


# --------------------------------------------------------------------------- #
# Triton kernel 层
# --------------------------------------------------------------------------- #
@requires_triton
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_kernel_matches_naive(dtype):
    q, k, v = _case(tq=64, tk=64, d=64, dtype=dtype, device="cuda")
    with torch.no_grad():
        got = flash_attention(q, k, v, is_causal=True)
    tol = 2e-2 if dtype == torch.float16 else 2e-4
    assert torch.allclose(got.float(), _naive(q, k, v), atol=tol, rtol=tol)


@requires_triton
@requires_cuda
def test_flash_kernel_gqa_and_window():
    q, k, v = _case(hq=8, hkv=2, tq=65, tk=129, d=64, device="cuda")
    with torch.no_grad():
        got = flash_attention(q, k, v, is_causal=True, window=16, sinks=4)
    assert torch.allclose(got.float(), _naive(q, k, v, window=16, sinks=4),
                          atol=2e-4, rtol=1e-4)


@requires_triton
@requires_cuda
def test_flash_kernel_with_offset():
    q_full, k, v = _case(tq=96, tk=96, d=64, device="cuda")
    ref = _naive(q_full, k, v)
    with torch.no_grad():
        for t in (0, 33, 95):
            got = flash_attention(q_full[:, :, t:t + 1], k, v, is_causal=True, offset=t)
            assert torch.allclose(got.float(), ref[:, :, t:t + 1], atol=2e-4, rtol=1e-4)


@requires_triton
@requires_cuda
def test_flash_matches_blocked_ref():
    """Triton kernel 必须和自己的 PyTorch 译文一致（这是最直接的回归测试）。"""
    q, k, v = _case(tq=70, tk=110, d=64, device="cuda")
    with torch.no_grad():
        got = flash_attention(q, k, v, is_causal=True)
    assert torch.allclose(got.float(),
                          flash_attention_blocked_ref(q, k, v, is_causal=True).float(),
                          atol=2e-4, rtol=1e-4)
