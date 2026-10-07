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
