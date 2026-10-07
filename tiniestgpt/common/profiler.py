"""性能剖析工具：计时、参数量、FLOPs 估算、MFU、显存、微基准。

MFU（Model FLOPs Utilization）是训练/推理系统的核心指标：
    MFU = 实际达成的 FLOPs/s ÷ 硬件峰值 FLOPs/s
只有把 FLOPs 算清楚，才知道"优化到底有没有用"。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

__all__ = [
    "Timer", "AvgMeter", "count_params", "transformer_flops_per_token",
    "MFUTracker", "benchmark", "cuda_memory_mb", "peak_memory_mb",
    "torch_profile", "profile_summary", "export_chrome_trace", "nvtx_range",
    "memory_snapshot", "nvtx_available",
]


class Timer:
    """可累加的计时器，支持 CUDA 同步与上下文管理器。"""

    def __init__(self, name: str = "", sync_cuda: bool = True, enabled: bool = True) -> None:
        self.name = name
        self.sync_cuda = sync_cuda
        self.enabled = enabled
        self.total = 0.0
        self.calls = 0
        self._start: Optional[float] = None

    def _now(self) -> float:
        if self.sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:  # pragma: no cover
                pass
        return time.perf_counter()

    def start(self) -> "Timer":
        if self.enabled:
            self._start = self._now()
        return self

    def stop(self) -> float:
        if not self.enabled or self._start is None:
            return 0.0
        dt = self._now() - self._start
        self.total += dt
        self.calls += 1
        self._start = None
        return dt

    @property
    def mean(self) -> float:
        return self.total / max(self.calls, 1)

    @contextmanager
    def time(self) -> Iterator["Timer"]:
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class AvgMeter:
    """滑动平均（可选窗口），用于 loss / 吞吐等指标。"""

    def __init__(self, window: int = 0) -> None:
        self.window = window
        self._vals: list[float] = []
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1) -> None:
        self._vals.append(v)
        if self.window and len(self._vals) > self.window:
            self._vals.pop(0)
        self.sum += v * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.window and self._vals:
            return sum(self._vals) / len(self._vals)
        return self.sum / max(self.count, 1)

    def reset(self) -> None:
        self._vals.clear()
        self.sum = 0.0
        self.count = 0


def count_params(model: "object", trainable_only: bool = False) -> Dict[str, int]:
    """统计参数量：总数 / 可训练数 / 按顶层模块拆分。"""
    import torch.nn as nn

    assert isinstance(model, nn.Module)
    total = trainable = 0
    by_module: Dict[str, int] = {}
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        top = name.split(".")[0] if "." in name else name
        by_module[top] = by_module.get(top, 0) + n
    return {"total": total, "trainable": trainable, "by_module": by_module}


def transformer_flops_per_token(
    n_params: int,
    n_layers: int,
    dim: int,
    seq_len: int,
    vocab_size: int,
    active_params: Optional[int] = None,
) -> Dict[str, float]:
    """估算每 token 的 FLOPs。

    经验公式（忽略 attention softmax 等小项）：
      * 前向 ≈ 2 * N   （一次矩阵乘 = 2 * W 的乘加）
      * 训练 ≈ 6 * N   （前向 2N + 反向 4N）
      * attention 额外 ≈ 4 * L * d * S（QK^T 与 AV 各 2*L*d*S）
    MoE 场景下用 ``active_params`` 替代 N。
    """
    n = active_params if active_params else n_params
    # 去掉 embedding 的"稠密计算"假设：embedding 是查表，不计入 FLOPs
    non_embedding = max(n - vocab_size * dim, 1)
    fwd = 2.0 * non_embedding
    attn = 4.0 * n_layers * dim * seq_len
    fwd_total = fwd + attn
    return {
        "forward_per_token": fwd_total,
        "forward_backward_per_token": 3.0 * fwd_total,
        "attention_per_token": attn,
    }


class MFUTracker:
    """累加 token 与耗时，输出 tokens/s 与 MFU。"""

    # 常见卡的 bf16/fp16 稠密峰值（FLOPs/s），可按需扩展
    PEAK_FLOPS: Dict[str, float] = {
        "A100": 312e12, "H100": 989e12, "H200": 989e12,
        "A800": 312e12, "4090": 165e12, "3090": 142e12,
        "3060": 25.6e12, "V100": 125e12, "T4": 65e12,
    }

    def __init__(self, flops_per_token: float, device_name: Optional[str] = None, peak_flops: Optional[float] = None) -> None:
        self.flops_per_token = flops_per_token
        self.tokens = 0
        self.seconds = 0.0
        self.peak_flops = peak_flops or self._detect_peak(device_name)

    def _detect_peak(self, device_name: Optional[str]) -> float:
        try:
            import torch

            if not torch.cuda.is_available():
                return float("nan")
            name = device_name or torch.cuda.get_device_name(0)
        except Exception:
            name = device_name or ""
        for k, v in self.PEAK_FLOPS.items():
            if k.lower() in name.lower().replace(" ", ""):
                return v
        return float("nan")

    def update(self, tokens: int, seconds: float) -> None:
        self.tokens += int(tokens)
        self.seconds += float(seconds)

    def reset(self) -> None:
        self.tokens = 0
        self.seconds = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / max(self.seconds, 1e-9)

    @property
    def flops_per_second(self) -> float:
        return self.tokens_per_second * self.flops_per_token

    @property
    def mfu(self) -> float:
        if self.peak_flops != self.peak_flops:  # NaN
            return float("nan")
        return self.flops_per_second / self.peak_flops

    def report(self) -> str:
        return (
            f"{self.tokens_per_second:,.0f} tok/s | "
            f"{self.flops_per_second / 1e12:,.1f} TFLOP/s | MFU={self.mfu:.1%}"
        )


def benchmark(fn: Callable[[], Any], warmup: int = 3, iters: int = 20, sync_cuda: bool = True) -> Dict[str, float]:
    """微基准：返回均值/标准差/最小值（毫秒）。"""
    import statistics

    def _sync() -> None:
        if sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:  # pragma: no cover
                pass

    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def nvtx_available() -> bool:
    try:
        import torch

        return hasattr(torch.cuda, "nvtx")
    except Exception:  # pragma: no cover
        return False


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Nsight Systems 里显示的自定义区间。

    在训练循环的关键段落（数据加载 / forward / backward / optimizer / 通信）
    各打一个 range，就能在 nsys 的时间线上一眼看出 GPU idle gap 是谁造成的。
    没有 nvtx 时是空操作。
    """
    push = pop = None
    try:
        import torch

        nvtx = getattr(torch.cuda, "nvtx", None)
        if nvtx is not None and hasattr(nvtx, "range_push"):
            push, pop = nvtx.range_push, nvtx.range_pop
    except Exception:  # pragma: no cover
        pass
    if push is None:
        yield
        return
    push(name)
    try:
        yield
    finally:
        pop()


@contextmanager
def torch_profile(log_dir: Optional[str] = None, top: int = 0,
                  record_shapes: bool = True, profile_memory: bool = True,
                  with_stack: bool = False) -> Iterator[Any]:
    """``torch.profiler`` 的便捷封装。

    ::

        with torch_profile("out/prof") as prof:
            for step, batch in enumerate(loader):
                train_step(batch)
                prof.step()
                if step >= 8:
                    break
        print(profile_summary(prof))

    :param log_dir: 给定时会把 chrome trace 写到该目录（可用 chrome://tracing 打开）
    :param top:     >0 时退出前直接打印耗时最高的 ``top`` 个算子
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    try:
        import torch

        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    except Exception:  # pragma: no cover
        pass

    handlers = []
    if log_dir:
        from torch.profiler import tensorboard_trace_handler

        handlers.append(tensorboard_trace_handler(log_dir))

    with profile(
        activities=activities,
        on_trace_ready=handlers[0] if handlers else None,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof

    if top:
        print(profile_summary(prof, top=top))


def profile_summary(prof: Any, top: int = 15, sort_by: str = "self_cuda_time_total") -> str:
    """按 CUDA 自身耗时排序的前 N 个算子（定位瓶颈的第一步）。"""
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=top)
    except Exception as exc:  # pragma: no cover
        return f"(profiler 输出失败: {exc})"


def export_chrome_trace(prof: Any, path: str) -> str:
    """导出 chrome trace（``chrome://tracing`` 或 Perfetto 打开）。"""
    prof.export_chrome_trace(path)
    return path


def memory_snapshot(path: str = "out/memory_snapshot.pkl") -> str:
    """记录显存分配历史，用来排查碎片与峰值来源。

    需要先用 ``torch.cuda.memory._record_memory_history(True)`` 开启记录，
    本函数只在结束时落盘。配合 ``_snapshot_pickled`` 的可视化页面使用。
    """
    import pickle

    import torch

    snap = torch.cuda.memory._snapshot() if hasattr(torch.cuda.memory, "_snapshot") else {}
    with open(path, "wb") as f:
        pickle.dump(snap, f)
    return path


def cuda_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except Exception:  # pragma: no cover
        pass
    return 0.0


def peak_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:  # pragma: no cover
        pass
    return 0.0
