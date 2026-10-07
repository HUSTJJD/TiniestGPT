"""手写 CUDA 内核的 JIT 编译加载器。

设计原则（与项目其余部分一致）：**能降级，绝不炸**。

* 没有 GPU / 没有 nvcc / 编译失败 → :func:`load` 返回 ``None``，
  :mod:`cuda_ops` 里的每个算子自动回退到等价的 PyTorch 实现。
* 编译结果由 torch 的扩展构建目录缓存（默认 ``~/.cache/torch_extensions``），
  只在本文件源码变化时重新编译。
* 全程惰性：import 本模块**不会**触发 nvcc。

手动编译/排错::

    python -c "from tiniestgpt.kernels import loader; print(loader.load(verbose=True))"
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

__all__ = ["EXT_NAME", "nvcc_available", "cuda_available", "load", "build_error", "source_files"]

EXT_NAME = "tiniestgpt_cuda_kernels"

_CSRC = Path(__file__).parent / "csrc"

# 顺序很重要：common.cuh 在最前（宏与 include），bindings.cu 在最后（导出符号）
SOURCE_FILES = (
    "common.cuh",
    "01_vector_add.cu",
    "02_reduce.cu",
    "03_gemm.cu",
    "04_softmax.cu",
    "05_transpose.cu",
    "bindings.cu",
)

_mod: Optional[object] = None
_error: str = ""
_lock = threading.Lock()
_needs_allow_unsupported = False


def source_files() -> list[Path]:
    return [_CSRC / name for name in SOURCE_FILES]


def nvcc_available() -> bool:
    """nvcc 是否在 PATH 里（决定能否 JIT 编译）。"""
    return shutil.which("nvcc") is not None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def build_error() -> str:
    """上一次编译失败的原因；成功时为空串。"""
    return _error


def _nvcc_max_msc_ver() -> int:
    """从 nvcc 自带头文件里读出它支持的 **最高** ``_MSC_VER``（不含）。

    ``crt/host_config.h`` 里写着 ``#if _MSC_VER < 1910 || _MSC_VER >= 1940``，
    硬编码这个值会在换 CUDA 版本时失效，所以直接从头文件解析。
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return 1940
    cfg = Path(nvcc).resolve().parent.parent / "include" / "crt" / "host_config.h"
    try:
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"_MSC_VER\s*<\s*\d+\s*\|\|\s*_MSC_VER\s*>=\s*(\d+)", text)
        if m:
            return int(m.group(1))
    except Exception:  # pragma: no cover
        pass
    return 1940


def _pick_msvc_toolset(install: Path, max_msc: int) -> tuple[str, bool]:
    """在已安装的 MSVC 工具集里挑一个 nvcc 认的（尽量新）。

    ``VC/Tools/MSVC/<ver>`` 的 ``<ver>`` 形如 ``14.29.30133``，
    对应的 ``_MSC_VER`` 是 ``1900 + minor``（14.29 → 1929，14.44 → 1944）。

    返回 ``(工具集名, 是否需要 -allow-unsupported-compiler)``。
    若所有已装工具集都比 nvcc 的白名单新，就退而求其次：选最新的那个 +
    请求 nvcc 放宽检查（不这么做连编译都进不去）。

    可用环境变量 ``TINIESTGPT_MSVC_TOOLSET=14.44`` 强制指定。
    """
    forced = os.environ.get("TINIESTGPT_MSVC_TOOLSET", "").strip()
    if forced:
        return forced, True

    root = install / "VC" / "Tools" / "MSVC"
    if not root.exists():
        return "", False
    cands: list[tuple[int, str]] = []
    for d in root.iterdir():
        parts = d.name.split(".")
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        cands.append((1900 + int(parts[1]), f"{parts[0]}.{parts[1]}"))
    if not cands:
        return "", False
    cands.sort(reverse=True)
    for msc, name in cands:
        if msc < max_msc:
            return name, False
    return cands[0][1], True


def _setup_msvc_env() -> str:
    r"""Windows 上自动把 MSVC 的编译环境注入 ``os.environ``。

    两个坑：

    1. ``cl.exe`` 通常只在"VS 开发者命令行"里才在 PATH 中 →
       用 ``vswhere`` 定位 VS，执行 ``vcvarsall.bat`` 后把环境变量搬进来；
    2. nvcc 对 host 编译器有**白名单**：CUDA 12.x 只认到 VS 2022（``_MSC_VER < 1940``），
       而 VS 2026 带的是 MSVC 14.5x（``_MSC_VER`` 1950+），会直接 ``#error``。
       强行加 ``-allow-unsupported-compiler`` 也只是让 ``cudafe++`` 崩溃，
       所以正确做法是**降级到已安装的旧工具集**（``-vcvars_ver=14.29``）。

    返回空串表示成功，否则返回失败原因（调用方据此降级）。
    """
    if os.name != "nt":
        return ""

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return "" if shutil.which("cl") else \
            "PATH 中没有 cl.exe，且找不到 vswhere.exe（需要安装 VS 生成工具）"

    try:
        proc = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=30)
        install_lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if not install_lines:
            return "vswhere 未返回任何 VS 安装路径"
        install = Path(install_lines[0])
        vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if not vcvars.exists():
            return f"找不到 {vcvars}"

        args = ["cmd", "/c", "call", str(vcvars), "x64"]
        toolset, needs_allow = _pick_msvc_toolset(install, _nvcc_max_msc_ver())
        if toolset:
            args.append(f"-vcvars_ver={toolset}")
        args += ["&&", "set"]
        global _needs_allow_unsupported
        _needs_allow_unsupported = needs_allow

        # 注意：必须以“参数列表”形式调用 cmd，拼成单个字符串时 cmd 会
        # 把引号一起当路径的一部分（表现为 "is not recognized"）。
        env = subprocess.run(args, capture_output=True, text=True,
                             timeout=120, errors="replace")
        for line in env.stdout.splitlines():
            if "=" not in line or line.startswith(("*", "[")):
                continue                       # 跳过 vcvarsall 的横幅
            k, v = line.split("=", 1)
            os.environ[k] = v
    except Exception as exc:  # pragma: no cover - 平台相关
        return f"注入 MSVC 环境失败：{exc}"

    return "" if shutil.which("cl") else "vcvarsall 执行后仍找不到 cl.exe"


def _ascii_safe(src: str) -> str:
    """剥离非 ASCII 字符，只保留代码。

    ``csrc/*.cu`` 里写满了中文注释，读源码时很爽；但 torch 的 ``load_inline``
    在 Windows 上用**系统默认编码（GBK）**把源码写回临时文件，遇到中文/emoji
    会直接抛 ``UnicodeEncodeError``。所以编译时把注释丢掉，只留 ASCII 代码。

    源码文件本身保持 UTF-8 不变——文档价值不打折。
    """
    out: list[str] = []
    for line in src.splitlines():
        if all(ord(c) < 128 for c in line):
            out.append(line)
            continue
        # 行尾注释：保留注释左边的代码（简单判断引号，避免误伤字符串里的 //）
        idx = line.find("//")
        if idx >= 0 and line.count('"', 0, idx) % 2 == 0:
            head = line[:idx].rstrip()
            out.append(head)
        else:
            out.append("")          # 整行都是中文注释 → 丢弃
    return "\n".join(out)


def _patch_decode_args() -> None:
    """把 torch 解码子进程输出的方式改成 UTF-8（Windows 专用补丁）。

    中文 Windows 上 ``cl.exe`` 会输出 **UTF-8** 的用法提示（"用法: cl ..."），
    而 torch 默认用 ``('oem',)``（GBK）解码 → ``UnicodeDecodeError: 'cp1' ...``，
    直接让 JIT 编译失败。改成 ``('utf-8', 'replace')`` 后既能读到版本号，
    也不会因为个别不可解字节而中断。
    """
    if os.name != "nt":
        return
    try:
        from torch.utils import cpp_extension as _ce

        setattr(_ce, "SUBPROCESS_DECODE_ARGS", ("utf-8", "replace"))
    except Exception:  # pragma: no cover
        pass


def _is_unsupported_host_compiler(exc: BaseException) -> bool:
    msg = str(exc)
    return "unsupported Microsoft Visual Studio version" in msg or "unsupported compiler" in msg


def _combined_source() -> str:
    return "\n".join(_ascii_safe(p.read_text(encoding="utf-8")) for p in source_files())


def load(force_rebuild: bool = False, verbose: bool = False) -> Optional[object]:
    """编译（或复用缓存）并返回扩展模块；不可用时返回 ``None``。"""
    global _mod, _error
    if _mod is not None and not force_rebuild:
        return _mod

    with _lock:
        if _mod is not None and not force_rebuild:
            return _mod

        if not cuda_available():
            _error = "torch.cuda.is_available() == False"
            return None
        if not nvcc_available():
            _error = "PATH 中找不到 nvcc（需要安装 CUDA Toolkit）"
            return None

        msvc_err = _setup_msvc_env()
        if msvc_err:
            _error = msvc_err
            return None

        try:
            import torch
            from torch.utils.cpp_extension import load_inline

            _patch_decode_args()
            torch.cuda.init()      # 触发 lazy 初始化，避免 nvcc 拿不到 arch

            src = _combined_source()
            # -lineinfo 供 Nsight Compute 下钻；-allow-unsupported-compiler 仅在
            # 本机 MSVC 比 nvcc 白名单还新时才加（见 _pick_msvc_toolset）。
            base_flags = ["-O3", "-lineinfo"]
            if _needs_allow_unsupported or os.environ.get("TINIESTGPT_NVCC_ALLOW_UNSUPPORTED"):
                base_flags.append("-allow-unsupported-compiler")

            def _compile(flags: list[str]) -> object:
                return load_inline(
                    name=EXT_NAME,
                    cpp_sources=["// 所有实现都在 .cu 里，此处仅需一个空的 host 编译单元"],
                    cuda_sources=[src],
                    functions=None,               # 绑定写在 csrc/bindings.cu 里
                    extra_cuda_cflags=flags,
                    verbose=verbose,
                    is_python_module=True,
                )

            try:
                _mod = _compile(base_flags)
            except Exception as exc:
                # nvcc 会对 host 编译器做白名单校验：CUDA 12.x 只认到 VS 2022，
                # 而 VS 2026（MSVC 19.5x）会被判为 "unsupported"。
                # 教学用的 kernel 不碰 MSVC 特性，加该标志是安全的。
                if not _is_unsupported_host_compiler(exc):
                    raise
                _mod = _compile(base_flags + ["-allow-unsupported-compiler"])
            _error = ""
        except Exception as exc:  # pragma: no cover - 平台相关
            _mod = None
            _error = f"{type(exc).__name__}: {exc}"
    return _mod
