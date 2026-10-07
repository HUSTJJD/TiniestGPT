"""推理侧增强：Radix 前缀缓存 / CPU 交换空间 / INT4 融合 GEMM。"""

from __future__ import annotations

import torch
import torch.nn as nn

from tiniestgpt.inference.kernels.int4_gemm import (
    TRITON_AVAILABLE, fused_int4_gemm, fused_int4_gemm_torch, int4_gemm_ref,
)
from tiniestgpt.inference.quantization.base import QuantizedLinear
from tiniestgpt.inference.radix_cache import RadixPrefixCache
from tiniestgpt.inference.swap import CPUSwapSpace


# --------------------------------------------------------------------------- #
# Radix 前缀缓存
# --------------------------------------------------------------------------- #
def test_radix_miss_then_hit():
    cache = RadixPrefixCache(block_size=4)
    prompt = list(range(16))
    assert cache.match_prefix(prompt)[1] == 0            # 未插入 → 未命中
    cache.insert(prompt, [10, 11, 12, 13])
    blocks, matched = cache.match_prefix(prompt)
    assert matched == 16
    assert blocks == [10, 11, 12, 13]
    assert cache.hit_rate > 0


def test_radix_partial_match_and_fork():
    """两条 prompt 共享前 8 个 token → 第二条只命中前两块。"""
    cache = RadixPrefixCache(block_size=4)
    p1 = list(range(16))
    p2 = list(range(8)) + [100, 101, 102, 103, 104, 105, 106, 107]
    cache.insert(p1, [1, 2, 3, 4])
    blocks, matched = cache.match_prefix(p2)
    assert matched == 8
    assert blocks == [1, 2]


def test_radix_disabled_is_noop():
    cache = RadixPrefixCache(block_size=4, enabled=False)
    cache.insert(list(range(16)), [1, 2, 3, 4])
    assert cache.match_prefix(list(range(16))) == ([], 0)


def test_radix_eviction_frees_blocks_lru():
    cache = RadixPrefixCache(block_size=4)
    cache.insert(list(range(16)), [1, 2, 3, 4])          # 较早插入
    cache.insert([99] * 16, [5, 6, 7, 8])                # 较晚插入 → 更"热"
    freed = cache.evict(4)
    assert len(freed) == 4
    assert set(freed) == {1, 2, 3, 4}                    # LRU：先淘汰最久未访问的
    assert cache.stats()["evictions"] == 4


def test_radix_eviction_skips_referenced_nodes():
    cache = RadixPrefixCache(block_size=4)
    cache.insert(list(range(16)), [1, 2, 3, 4])

    def mark(n):                                         # 整条前缀都被某个请求引用
        n.refcount = 1
        for c in n.children.values():
            mark(c)

    mark(cache.root)
    assert cache.evict(4) == []


# --------------------------------------------------------------------------- #
# CPU 交换空间
# --------------------------------------------------------------------------- #
def test_swap_roundtrip_preserves_data():
    torch.manual_seed(0)
    n_layers, num_blocks, bs, hkv, d = 2, 8, 4, 2, 8
    k = torch.randn(n_layers, num_blocks, bs, hkv, d)
    v = torch.randn(n_layers, num_blocks, bs, hkv, d)
    space = CPUSwapSpace(k, v, num_blocks=4)

    slots = space.swap_out([2, 5])
    assert len(slots) == 2
    assert space.num_free == 2
    # 污染 GPU 上的块，验证 swap_in 能把原数据搬回来
    k[:, 2] = -1.0
    v[:, 5] = -1.0
    space.swap_in(slots, [2, 5])
    assert torch.allclose(k[:, 2], space.k[slots[0]])
    assert torch.allclose(v[:, 5], space.v[slots[1]])
    assert space.num_free == 4
    assert space.stats["swap_out_blocks"] == 2
    assert space.stats["swap_in_blocks"] == 2


def test_swap_space_exhaustion_raises():
    k = torch.zeros(1, 4, 2, 1, 1)
    space = CPUSwapSpace(k, k.clone(), num_blocks=2)
    space.swap_out([0, 1])
    try:
        space.swap_out([2])
        raise AssertionError("应当抛出空间不足")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# INT4 融合 GEMM
# --------------------------------------------------------------------------- #
def _quantized(bits: int = 4, group_size: int = 32, out_f: int = 64, in_f: int = 128):
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f, bias=True)
    return QuantizedLinear.from_float(lin, bits=bits, group_size=group_size)


def test_fused_int4_torch_matches_default_forward():
    q = _quantized()
    x = torch.randn(5, 128)
    ref = nn.functional.linear(x, q.dequantize_weight().to(x.dtype), q.bias)
    got = fused_int4_gemm_torch(q.qweight, q.scales, x, q.group_size, q.q_shape, q.bias)
    assert torch.allclose(got, ref, atol=1e-4)


def test_int4_gemm_ref_matches_dequantized():
    q = _quantized()
    x = torch.randn(3, 128)
    ref = nn.functional.linear(x, q.dequantize_weight().to(x.dtype), None)
    got = int4_gemm_ref(q.qweight, q.scales, x, q.group_size, q.q_shape)
    assert torch.allclose(got, ref, atol=1e-4)


def test_fused_int4_gemm_falls_back_on_cpu():
    """没有 Triton 时 fused_int4_gemm 必须给出与参考实现一致的结果。"""
    q = _quantized()
    x = torch.randn(4, 128)
    ref = nn.functional.linear(x, q.dequantize_weight().to(x.dtype), q.bias)
    got = fused_int4_gemm(q.qweight, q.scales, x, q.group_size, q.q_shape, q.bias)
    assert torch.allclose(got, ref, atol=1e-4)


def test_quantized_linear_uses_fused_path_when_enabled():
    q = _quantized()
    q.use_fused_kernel = True           # 本机会因为没有 Triton 自动回退到普通路径
    x = torch.randn(2, 128)
    ref = nn.functional.linear(x, q.dequantize_weight().to(x.dtype), q.bias)
    assert torch.allclose(q(x), ref, atol=1e-4)


def test_int4_quantization_error_is_bounded():
    """融合路径不能放大量化误差：与 fp32 原权重的相对误差应在同一量级。"""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=False)
    q = QuantizedLinear.from_float(lin, bits=4, group_size=32)
    x = torch.randn(8, 128)
    ref = lin(x)
    err_ref = (ref - nn.functional.linear(x, q.dequantize_weight().to(x.dtype))).abs().max()
    err_fused = (ref - fused_int4_gemm_torch(q.qweight, q.scales, x, q.group_size, q.q_shape)).abs().max()
    assert err_fused <= err_ref + 1e-5


def _triton_skip():
    if not (TRITON_AVAILABLE and torch.cuda.is_available()):
        return True
    return False


def test_fused_int4_kernel_on_gpu_if_available():
    if _triton_skip():
        import pytest

        pytest.skip("需要 Triton + CUDA（仅 Linux）")
    q = _quantized().cuda()
    x = torch.randn(16, 128, device="cuda", dtype=torch.float16)
    ref = nn.functional.linear(x, q.dequantize_weight().to(x.dtype), q.bias)
    got = fused_int4_gemm(q.qweight, q.scales, x, q.group_size, q.q_shape, q.bias)
    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2)
