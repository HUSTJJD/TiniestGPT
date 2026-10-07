"""性能剖析入口：torch.profiler / Nsight Systems / Nsight Compute 三条工具链。

对应 AIInfraGuide 路线 3.6「性能分析与 Benchmark」与 1.2 的工具链：

| 工具 | 回答什么 | 本脚本用法 |
|---|---|---|
| ``torch.profiler`` | 哪一步慢、有没有 CPU-GPU 空洞 | ``profile.py torch`` |
| Nsight Systems | GPU idle gap 来自 CPU / 通信 / launch 开销 | ``profile.py nsys -- <命令>`` |
| Nsight Compute | 单个 kernel 是 memory bound 还是 compute bound | ``profile.py ncu -- <命令>`` |

示例::

    python scripts/profile.py torch --config recipes/pretrain_tiny.yaml --steps 6
    python scripts/profile.py nsys -- python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml
    python scripts/profile.py ncu  -- python benchmarks/cuda_kernels.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

__all__ = ["main"]


def _run_torch_profiler(config: str, steps: int, log_dir: str, top: int,
                        batch_size: int, seq_len: int) -> int:
    import torch

    from tiniestgpt.common.config import load_config
    from tiniestgpt.common.profiler import MFUTracker, count_params, export_chrome_trace, \
        nvtx_range, profile_summary, torch_profile, transformer_flops_per_token
    from tiniestgpt.model.factory import build_model
    from tiniestgpt.train.config import TrainConfig

    cfg = load_config(TrainConfig, config)
    model_cfg = getattr(cfg, "model", cfg)
    model = build_model(model_cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()

    n_params = count_params(model)["total"]
    print(f"模型参数量: {n_params:,}   设备: {dev}")

    flops = transformer_flops_per_token(
        n_params, model.cfg.n_layers, model.cfg.dim, seq_len, model.cfg.vocab_size)
    mfu = MFUTracker(flops["forward_backward_per_token"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=dev)
    y = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=dev)

    os.makedirs(log_dir, exist_ok=True)
    trace_path = str(Path(log_dir) / "trace.json")

    # 先 warmup（避开 CUDA 首次初始化与 cudnn autotune）
    def _step() -> None:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, model.cfg.vocab_size), y.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):
        _step()
    if dev == "cuda":
        torch.cuda.synchronize()

    with torch_profile(log_dir=log_dir) as prof:
        t0 = time.perf_counter()
        for step in range(steps):
            with nvtx_range(f"step_{step}"):
                with nvtx_range("forward"):
                    logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, model.cfg.vocab_size), y.view(-1))
                with nvtx_range("backward"):
                    loss.backward()
                with nvtx_range("optimizer"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            mfu.update(batch_size * seq_len, time.perf_counter() - t0)
            t0 = time.perf_counter()
            prof.step()

    print()
    print(profile_summary(prof, top=top))
    traces = sorted(Path(log_dir).glob("*.trace.json"))
    if traces:
        # tensorboard_trace_handler 已经落盘，直接指给用户
        print(f"chrome trace: {traces[-1]}")
        print("打开方式：chrome://tracing 加载，或直接拖进 https://ui.perfetto.dev")
    else:  # pragma: no cover
        try:
            print(f"chrome trace 已导出: {export_chrome_trace(prof, trace_path)}")
        except Exception as exc:
            print(f"导出 trace 失败: {exc}")
    print()
    print("吞吐:", mfu.report())
    if dev == "cuda":
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB")
    return 0


def _run_external(tool: str, cmd: list[str], out_dir: str, extra: list[str]) -> int:
    exe = shutil.which(tool)
    if not exe:
        tips = {
            "nsys": "Nsight Systems 未安装：https://developer.nvidia.com/nsight-systems",
            "ncu": "Nsight Compute 未安装：https://developer.nvidia.com/nsight-compute",
        }
        print(tips.get(tool, f"找不到 {tool}"))
        return 2
    os.makedirs(out_dir, exist_ok=True)
    name = f"{tool}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = str(Path(out_dir) / name)

    if tool == "nsys":
        args = [exe, "profile", "-o", out, "--trace=cuda,nvtx,osrt,cudnn", *extra, "--"] + cmd
    else:
        args = [exe, "--set", "full", "-o", out, "--target-processes", "all", *extra] + cmd

    print("执行:", " ".join(args))
    r = subprocess.run(args)
    if r.returncode != 0:
        return r.returncode
    suffix = ".nsys-rep" if tool == "nsys" else ".ncu-rep"
    print(f"\n报告: {out}{suffix}")
    print("打开方式:", f"nsys-ui {out}{suffix}" if tool == "nsys" else f"ncu-ui {out}{suffix}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TiniestGPT 性能剖析入口")
    sub = ap.add_subparsers(dest="tool", required=True)

    p1 = sub.add_parser("torch", help="用 torch.profiler 剖析若干训练步")
    p1.add_argument("--config", default="recipes/pretrain_tiny.yaml")
    p1.add_argument("--steps", type=int, default=6)
    p1.add_argument("--batch-size", type=int, default=8)
    p1.add_argument("--seq-len", type=int, default=256)
    p1.add_argument("--log-dir", default="out/prof")
    p1.add_argument("--top", type=int, default=15)

    for name, help_text in (("nsys", "Nsight Systems：CPU-GPU 全链路时间线"),
                            ("ncu", "Nsight Compute：kernel 级下钻")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--out-dir", default=f"out/{name}")
        p.add_argument("cmd", nargs=argparse.REMAINDER,
                       help="要剖析的命令，用 -- 与前面的参数隔开")

    args = ap.parse_args(argv)

    if args.tool == "torch":
        return _run_torch_profiler(args.config, args.steps, args.log_dir, args.top,
                                   args.batch_size, args.seq_len)

    cmd = [c for c in getattr(args, "cmd", []) if c != "--"]
    if not cmd:
        print("请在 -- 之后给出要剖析的命令，例如：")
        print("  python scripts/profile.py nsys -- python -m tiniestgpt.cli pretrain "
              "--config recipes/pretrain_tiny.yaml")
        return 2
    return _run_external(args.tool, cmd, args.out_dir, [])


if __name__ == "__main__":
    sys.exit(main())
