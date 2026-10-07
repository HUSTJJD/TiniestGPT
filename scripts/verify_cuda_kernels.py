"""脱离 PyTorch 直接编译并自检手写 CUDA 内核。

为什么需要这条路径：
torch 的 JIT 扩展要求 "torch 头文件 ↔ nvcc ↔ MSVC" 三者版本互相兼容，
而这在很多机器上难以同时满足（例如 CUDA 12.3 + VS 2026 的 MSVC 19.5x）。
``csrc/*.cu`` 里的 kernel 本身并不依赖 PyTorch，所以在 ``-DTG_STANDALONE`` 下
可以单独编译出一个可执行程序做数值自检。

用法::

    python scripts/verify_cuda_kernels.py
    python scripts/verify_cuda_kernels.py --arch 86 --keep
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiniestgpt.kernels import loader  # noqa: E402


def _detect_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return os.environ.get("TINIESTGPT_CUDA_ARCH", "86")


def main() -> int:
    ap = argparse.ArgumentParser(description="编译并自检 TiniestGPT 手写 CUDA 内核")
    ap.add_argument("--arch", default=None, help="GPU 计算能力，如 86 / 89 / 90（默认自动探测）")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（便于看编译产物）")
    args = ap.parse_args()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("PATH 中找不到 nvcc，无法编译。请安装 CUDA Toolkit。")
        return 2

    # Windows 上把 MSVC 环境准备好（会挑一个 nvcc 认的工具集）
    msvc_err = loader._setup_msvc_env()
    if msvc_err:
        print(f"准备 MSVC 环境失败：{msvc_err}")
        return 2

    arch = (args.arch or _detect_arch()).replace(".", "")
    workdir = Path(tempfile.mkdtemp(prefix="tiniestgpt_cuda_"))
    try:
        parts = [loader._ascii_safe(p.read_text(encoding="utf-8")) for p in loader.source_files()]
        parts.append(loader._ascii_safe(
            (loader._CSRC / "standalone_main.cu").read_text(encoding="utf-8")))
        src = workdir / "kernels.cu"
        src.write_text("\n".join(parts), encoding="ascii")

        exe_name = "kernels_selfcheck.exe" if os.name == "nt" else "kernels_selfcheck"
        exe = workdir / exe_name
        cmd = [
            nvcc, "-O3", "-DTG_STANDALONE",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "-o", str(exe), str(src),
        ]
        print("编译:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(r.stdout[-4000:])
            print(r.stderr[-4000:], file=sys.stderr)
            print("\n编译失败。常见原因：nvcc 与 MSVC 版本不兼容（见 docs/08-cuda.md 第 6 节）。")
            return 1

        print("运行自检:")
        run = subprocess.run([str(exe)], capture_output=True, text=True, errors="replace")
        print(run.stdout)
        if run.stderr.strip():
            print(run.stderr, file=sys.stderr)
        return run.returncode
    finally:
        if args.keep:
            print(f"临时目录保留在: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
