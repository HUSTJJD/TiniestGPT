"""手写 CUDA 内核的 Python 封装：**每个算子都有 PyTorch 参考实现**。

这是本项目的一贯设计（见 ``model/kernels.py``）：先看懂 ``*_ref``，
再对比 CUDA 版本"到底快在哪"。没有 GPU / 编译失败时自动走参考实现，
因此**调用方永远不需要写 if**。

::

    from tiniestgpt.kernels import cuda_ops as ck
    print(ck.available())                       # 是否真的用上了 CUDA 内核
    print(ck.reduce_sum(x, version=2).item())   # v0/v1/v2 三个版本
    print(ck.benchmark_reduce(1 << 20))         # 三连对比（路线 1.3 的检验标准）
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from . import loader

__all__ = [
    "available", "available_ops", "status", "build_error",
    "vector_add", "reduce_sum", "gemm", "softmax", "transpose",
    "vector_add_ref", "reduce_sum_ref", "gemm_ref", "softmax_ref", "transpose_ref",
    "benchmark_reduce", "benchmark_gemm", "benchmark_softmax",
]


def _ext():
    return loader.load()


def available() -> bool:
    """CUDA 内核是否可用（GPU + nvcc + 编译成功三者缺一不可）。"""
    return _ext() is not None


def build_error() -> str:
    return loader.build_error()


def status() -> Dict[str, object]:
    return {
        "cuda_device": loader.cuda_available(),
        "nvcc": loader.nvcc_available(),
        "extension": available(),
        "error": build_error(),
    }


def available_ops() -> Dict[str, bool]:
    ok = available()
    return {name: ok for name in ("vector_add", "reduce_sum", "gemm", "softmax", "transpose")}


# --------------------------------------------------------------------------- #
# 参考实现（永远可用）
# --------------------------------------------------------------------------- #
def vector_add_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def reduce_sum_ref(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).sum()


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def transpose_ref(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).contiguous()


# --------------------------------------------------------------------------- #
# CUDA 实现（不可用时回退）
# --------------------------------------------------------------------------- #
def _usable(x: torch.Tensor, cuda: bool) -> bool:
    return cuda and x.device.type == "cuda" and x.dtype == torch.float32 and available()


def vector_add(a: torch.Tensor, b: torch.Tensor, use_cuda: bool = True) -> torch.Tensor:
    if not _usable(a, use_cuda) or b.device != a.device or b.dtype != a.dtype:
        return vector_add_ref(a, b)
    return _ext().vector_add(a.contiguous(), b.contiguous())


def reduce_sum(x: torch.Tensor, version: int = 2, use_cuda: bool = True) -> torch.Tensor:
    """0=原子加 / 1=共享内存树形 / 2=warp shuffle。"""
    flat = x.reshape(-1)
    if not _usable(flat, use_cuda) or not flat.is_contiguous():
        return reduce_sum_ref(flat)
    return _ext().reduce_sum(flat, int(version))


def gemm(a: torch.Tensor, b: torch.Tensor, tiled: bool = True, use_cuda: bool = True) -> torch.Tensor:
    """[M,K] @ [K,N]。CUDA 路径只支持 2D float32 连续张量。"""
    if a.dim() != 2 or b.dim() != 2 or not _usable(a, use_cuda) or b.dtype != a.dtype:
        return gemm_ref(a, b)
    return _ext().gemm(a.contiguous(), b.contiguous(), int(tiled))


def softmax(x: torch.Tensor, online: bool = True, use_cuda: bool = True) -> torch.Tensor:
    """行 softmax。online=True 走单趟 online normalizer。"""
    flat = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
    if not _usable(flat, use_cuda):
        return softmax_ref(x)
    out = _ext().softmax(flat.contiguous(), int(online))
    return out.reshape(x.shape) if x.dim() != 2 else out


def transpose(x: torch.Tensor, padded: bool = True, use_cuda: bool = True) -> torch.Tensor:
    """2D 转置。padded=True 用 +1 列 padding 消除 bank conflict。"""
    if x.dim() != 2 or not _usable(x, use_cuda):
        return transpose_ref(x)
    return _ext().transpose(x.contiguous(), int(padded))


# --------------------------------------------------------------------------- #
# 微基准：把"优化省在哪里"量化出来
# --------------------------------------------------------------------------- #
def _bench(fn, warmup: int = 3, iters: int = 30) -> Dict[str, float]:
    import statistics

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        for _ in range(warmup):
            fn()
        sync()
        ts = []
        for _ in range(iters):
            sync()
            t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if t0 is not None:
                t0.record()
                fn()
                t1 = torch.cuda.Event(enable_timing=True)
                t1.record()
                sync()
                ts.append(t0.elapsed_time(t1))
            else:  # pragma: no cover
                import time

                s = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - s) * 1000.0)
        return {"mean_ms": statistics.fmean(ts), "min_ms": min(ts)}
    except Exception as exc:  # pragma: no cover
        return {"mean_ms": float("nan"), "min_ms": float("nan"), "error": str(exc)}


def benchmark_reduce(n: int = 1 << 20, device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """Reduce 三连：v0 原子加 → v1 共享内存 → v2 warp shuffle。"""
    if device == "cuda" and not torch.cuda.is_available():
        return {}
    x = torch.randn(n, device=device, dtype=torch.float32)
    out: Dict[str, Dict[str, float]] = {}
    for v in (0, 1, 2):
        out[f"v{v}"] = _bench(lambda v=v: reduce_sum(x, version=v))
    out["torch"] = _bench(lambda: x.sum())
    ref = float(reduce_sum_ref(x))
    out["max_abs_err"] = {"mean_ms": max(abs(float(reduce_sum(x, version=v)) - ref) for v in (0, 1, 2))}
    return out


def benchmark_gemm(m: int = 512, k: int = 512, n: int = 512, device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """朴素 GEMM vs 共享内存分块 vs cuBLAS（torch.mm）。"""
    if device == "cuda" and not torch.cuda.is_available():
        return {}
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    out = {
        "naive": _bench(lambda: gemm(a, b, tiled=False)),
        "tiled": _bench(lambda: gemm(a, b, tiled=True)),
        "cublas": _bench(lambda: torch.mm(a, b)),
    }
    ref = gemm_ref(a, b)
    out["max_abs_err"] = {"mean_ms": float((gemm(a, b, tiled=True) - ref).abs().max())}
    tiled_ms = out["tiled"]["mean_ms"]
    flops = 2.0 * m * k * n
    out["tiled_tflops"] = {"mean_ms": flops / (tiled_ms * 1e-3) / 1e12}
    out["cublas_tflops"] = {"mean_ms": flops / (out["cublas"]["mean_ms"] * 1e-3) / 1e12}
    return out


def benchmark_softmax(rows: int = 512, cols: int = 1024, device: str = "cuda") -> Dict[str, Dict[str, float]]:
    if device == "cuda" and not torch.cuda.is_available():
        return {}
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)
    out = {
        "naive": _bench(lambda: softmax(x, online=False)),
        "online": _bench(lambda: softmax(x, online=True)),
        "torch": _bench(lambda: torch.softmax(x, dim=-1)),
    }
    ref = softmax_ref(x)
    out["max_abs_err"] = {"mean_ms": float((softmax(x, online=True) - ref).abs().max())}
    return out
