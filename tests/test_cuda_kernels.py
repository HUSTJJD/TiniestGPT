"""手写 CUDA 内核的数值一致性测试。

没有 GPU / 没有 nvcc / 编译失败时，与 CUDA 相关的用例会整体 skip
（PyTorch 参考实现部分仍然会跑），因此本文件在任何环境下都不会让 CI 变红。
"""

from __future__ import annotations

import pytest
import torch

from tiniestgpt.kernels import cuda_ops as ck

_CUDA = ck.available()
_REASON = ck.build_error() or "CUDA 内核不可用"
requires_kernels = pytest.mark.skipif(not _CUDA, reason=_REASON)


# --------------------------------------------------------------------------- #
# 参考实现（任何环境都要能跑）
# --------------------------------------------------------------------------- #
def test_reference_impls():
    a = torch.randn(17)
    b = torch.randn(17)
    assert torch.allclose(ck.vector_add_ref(a, b), a + b)
    assert torch.allclose(ck.reduce_sum_ref(a), a.sum())
    x = torch.randn(4, 9)
    assert torch.allclose(ck.softmax_ref(x), torch.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(ck.transpose_ref(x), x.T)


def test_gemm_ref_matches_matmul():
    a = torch.randn(6, 11)
    b = torch.randn(11, 4)
    assert torch.allclose(ck.gemm_ref(a, b), a @ b, atol=1e-5)


def test_fallback_is_identical():
    """use_cuda=False 时必须与参考实现完全一致（降级路径不能改变语义）。"""
    a = torch.randn(32)
    b = torch.randn(32)
    assert torch.allclose(ck.vector_add(a, b, use_cuda=False), a + b)
    assert torch.allclose(ck.reduce_sum(a, version=2, use_cuda=False), a.sum())
    x = torch.randn(3, 8)
    assert torch.allclose(ck.softmax(x, use_cuda=False), torch.softmax(x, -1), atol=1e-6)


def test_status_never_raises():
    st = ck.status()
    assert set(st) >= {"cuda_device", "nvcc", "extension", "error"}
    assert isinstance(st["error"], str)


# --------------------------------------------------------------------------- #
# CUDA 内核（可用时才跑）
# --------------------------------------------------------------------------- #
@requires_kernels
def test_vector_add_kernel():
    a = torch.randn(100000, device="cuda")
    b = torch.randn(100000, device="cuda")
    assert torch.allclose(ck.vector_add(a, b), a + b, atol=1e-6)


@requires_kernels
@pytest.mark.parametrize("version", [0, 1, 2])
def test_reduce_three_ways(version: int):
    x = torch.randn(1 << 14, device="cuda")
    got = float(ck.reduce_sum(x, version=version))
    ref = float(x.sum())
    # v0（原子加）把 N 个数累加到同一个 float 上，误差随 N 增长，容差放宽
    tol = max(0.05, abs(ref) * (1e-3 if version == 0 else 1e-4))
    assert abs(got - ref) < tol


@requires_kernels
def test_reduce_versions_agree():
    """三个版本只是优化手段不同，结果必须一致（浮点误差内）。"""
    x = torch.randn(1 << 14, device="cuda")
    vals = [float(ck.reduce_sum(x, version=v)) for v in (0, 1, 2)]
    assert max(vals) - min(vals) < 0.05


@requires_kernels
@pytest.mark.parametrize("tiled", [False, True])
def test_gemm_vs_cublas(tiled: bool):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a = torch.randn(97, 128, device="cuda")
        b = torch.randn(128, 53, device="cuda")
        got = ck.gemm(a, b, tiled=tiled)
        assert torch.allclose(got, a @ b, atol=1e-3, rtol=1e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@requires_kernels
@pytest.mark.parametrize("online", [False, True])
def test_softmax_vs_torch(online: bool):
    x = torch.randn(37, 256, device="cuda") * 3.0
    got = ck.softmax(x, online=online)
    assert torch.allclose(got, torch.softmax(x, dim=-1), atol=1e-5)
    assert torch.allclose(got.sum(-1), torch.ones(37, device="cuda"), atol=1e-4)


@requires_kernels
@pytest.mark.parametrize("padded", [False, True])
def test_transpose_vs_torch(padded: bool):
    x = torch.randn(70, 130, device="cuda")
    got = ck.transpose(x, padded=padded)
    assert torch.equal(got, x.T.contiguous())


@requires_kernels
def test_benchmarks_run():
    r = ck.benchmark_reduce(1 << 16)
    assert set(r) >= {"v0", "v1", "v2", "torch"}
    g = ck.benchmark_gemm(128, 128, 128)
    assert set(g) >= {"naive", "tiled", "cublas"}
    s = ck.benchmark_softmax(64, 512)
    assert set(s) >= {"naive", "online", "torch"}
