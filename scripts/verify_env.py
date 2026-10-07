"""环境自检：确认 CUDA / 内核后端 / 依赖是否就绪。

    uv run python scripts/verify_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python      : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    try:
        import torch
    except ImportError:
        print("torch       : 未安装")
        return 1

    print(f"torch       : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda        : {cuda}")
    if cuda:
        print(f"  device    : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  vram      : {props.total_memory / 1024**3:.1f} GB")
        print(f"  sm        : sm_{props.major}{props.minor}")
        # 实测一次 bf16 矩阵乘，确认 CUDA 内核真的能跑
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        c = (a @ b).float().sum().item()
        print(f"  bf16 gemm : ok (sum={c:.2f})")
    else:
        print("  提示：CUDA 不可用。请确认已按 README 用 uv 从 PyTorch 官方索引安装 cu128 版本。")

    try:
        from tiniestgpt.model.dispatch import available_backends

        print(f"kernels     : {available_backends()}")
    except Exception as exc:  # pragma: no cover
        print(f"kernels     : 检查失败 ({exc})")

    # 手写 CUDA 内核：缺 nvcc / ninja / MSVC 任一都会优雅降级，这里给出原因
    try:
        from tiniestgpt.kernels import cuda_ops as ck

        st = ck.status()
        print(f"cuda kernels: {'可用' if st['extension'] else '不可用（自动回退 PyTorch 实现）'}"
              f"  [device={st['cuda_device']}, nvcc={st['nvcc']}]")
        if not st["extension"] and st["error"]:
            lines = st["error"].splitlines()
            # 编译日志里真正有用的一般是带 "error" 的那行，而不是 nvcc 命令行
            reason = next((l for l in lines if "error" in l.lower()), lines[0])
            print(f"  原因      : {reason[:160]}")
            print("  独立自检  : python scripts/verify_cuda_kernels.py  （不需要 torch 头文件）")
    except Exception as exc:  # pragma: no cover
        print(f"cuda kernels: 检查失败 ({exc})")

    try:
        from tiniestgpt.model.factory import build_model

        m = build_model("tiny")
        print(f"model       : {m.num_params()['total'] / 1e6:.1f} M params, "
              f"kv/token {m.kv_cache_bytes_per_token() / 1024:.2f} KB @fp16")
    except Exception as exc:  # pragma: no cover
        print(f"model       : 检查失败 ({exc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
