"""CUDA 内核基准：把"每一步优化到底省在哪里"量化出来。

对齐 AIInfraGuide 路线 1.3 的检验标准：
  * Reduce 三连：原子加 → 共享内存树形 → warp shuffle，对比 torch.sum
  * GEMM：朴素 → 共享内存分块 → cuBLAS（并给出达到 cuBLAS 的百分比）
  * Softmax：朴素 → online → torch.softmax
  * 转置：有/无 padding 的 bank conflict 对比

用法::

    python benchmarks/cuda_kernels.py
    python benchmarks/cuda_kernels.py --reduce-n 4194304 --gemm 1024 1024 1024
"""

from __future__ import annotations

import argparse
import json
import sys

from tiniestgpt.kernels import cuda_ops as ck


def _fmt_table(title: str, result: dict, note: str = "") -> str:
    lines = [f"\n=== {title} ==="]
    if note:
        lines.append(note)
    for k, v in result.items():
        if not isinstance(v, dict):
            lines.append(f"  {k:<12} {v}")
            continue
        if "mean_ms" in v and "min_ms" in v:
            lines.append(f"  {k:<12} mean={v['mean_ms']:8.3f} ms   min={v['min_ms']:8.3f} ms")
        else:
            lines.append(f"  {k:<12} {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TiniestGPT CUDA kernel benchmarks")
    ap.add_argument("--reduce-n", type=int, default=1 << 22, help="reduce 的长度")
    ap.add_argument("--gemm", type=int, nargs=3, default=[512, 512, 512], metavar=("M", "K", "N"))
    ap.add_argument("--softmax", type=int, nargs=2, default=[512, 1024], metavar=("ROWS", "COLS"))
    ap.add_argument("--json", action="store_true", help="以 JSON 输出（便于回归门禁）")
    args = ap.parse_args(argv)

    st = ck.status()
    if not st["extension"]:
        print("CUDA 内核不可用，本次只输出环境诊断（不会报错）：")
        print(json.dumps(st, indent=2, ensure_ascii=False))
        print("\n排查顺序：① torch.cuda.is_available() ② nvcc 是否在 PATH ③ 首次编译日志")
        return 0

    out: dict = {"status": st}

    r = ck.benchmark_reduce(args.reduce_n)
    out["reduce"] = r
    if not args.json:
        print(_fmt_table(f"Reduce Sum (n={args.reduce_n:,})", r,
                         "v0 原子加 → v1 共享内存树形 → v2 warp shuffle"))

    m, k, n = args.gemm
    g = ck.benchmark_gemm(m, k, n)
    out["gemm"] = g
    if not args.json:
        ratio = g["tiled"]["mean_ms"] and g["cublas"]["mean_ms"] / g["tiled"]["mean_ms"]
        print(_fmt_table(f"GEMM ({m}x{k}x{n})", g,
                         f"分块版达到 cuBLAS 的 {ratio:.1%}（检验标准：≥50% 合格）"))

    rows, cols = args.softmax
    s = ck.benchmark_softmax(rows, cols)
    out["softmax"] = s
    if not args.json:
        print(_fmt_table(f"Softmax ({rows}x{cols})", s))

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
