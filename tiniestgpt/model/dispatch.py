"""内核后端分发：让"算法"与"实现"解耦。

注册表里的每个后端名字对应一个函数。默认 ``ref``（纯 PyTorch，永远可用）；
``inference/kernels/triton_kernels.py`` 在 import 时会把 ``triton`` 注册进来。
这样在 Windows / CPU / 无 Triton 环境下系统自动降级，在 Linux + GPU 上自动加速。
"""

from __future__ import annotations

import os
from typing import Callable, Dict

from .kernels import paged_attention_ref, sdpa_attention

__all__ = [
    "register_dense_attention", "register_paged_attention",
    "get_dense_attention", "get_paged_attention", "available_backends", "set_backend",
]

_DENSE: Dict[str, Callable] = {"ref": sdpa_attention, "sdpa": sdpa_attention}
_PAGED: Dict[str, Callable] = {"ref": paged_attention_ref}

_DEFAULT = os.environ.get("TINIESTGPT_ATTN_BACKEND", "auto")
_triton_attempted = False


def register_dense_attention(name: str, fn: Callable) -> None:
    _DENSE[name] = fn


def register_paged_attention(name: str, fn: Callable) -> None:
    _PAGED[name] = fn


def _try_load_triton() -> None:
    """延迟导入 Triton 内核（避免在 model 层硬依赖 inference 层造成循环导入）。"""
    global _triton_attempted
    if _triton_attempted:
        return
    _triton_attempted = True
    try:
        from ..inference.kernels.triton_kernels import register_triton_kernels

        register_triton_kernels()
    except Exception:          # Triton 缺失 / 平台不支持 / 编译失败 → 静默降级
        pass


def _resolve(table: Dict[str, Callable], backend: str) -> Callable:
    if backend in ("auto", ""):
        _try_load_triton()
        # auto 顺序里 flash 排在 sdpa 之后：FlashAttention 教学实现走 fp32/ieee
        # 精度且没有反向，默认不启用；想用请显式指定 attn_backend="flash"。
        order = ("triton", "sdpa", "flash", "ref")
    else:
        order = (backend, "ref")
    for name in order:
        if name in table:
            return table[name]
    return table["ref"]


def get_dense_attention(backend: str = "auto") -> Callable:
    return _resolve(_DENSE, backend)


def get_paged_attention(backend: str = "auto") -> Callable:
    return _resolve(_PAGED, backend)


def available_backends() -> Dict[str, list]:
    _try_load_triton()
    return {"dense": sorted(_DENSE), "paged": sorted(_PAGED)}


def set_backend(name: str) -> None:
    global _DEFAULT
    _DEFAULT = name
