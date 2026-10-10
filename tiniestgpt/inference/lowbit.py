"""微缩放低比特格式：NVFP4 / MXFP4（2026 的 4-bit 主力）。

**FP4、NVFP4、MXFP4、INT4 不是同一种东西**。它们都是"4 bit"，
但编码、scale 粒度、硬件指令、kernel 布局全都不同：

| 格式 | scale 粒度 | 代表模型 | 硬件 |
|---|---|---|---|
| NVFP4 | 每 16 个元素一个 E4M3 scale | Nemotron 3 Ultra、Mistral Small 4 | Blackwell |
| MXFP4 | 每 32 个元素共享一个 2 的幂 scale | gpt-oss | Blackwell / CDNA4 |
| FP4 (分组) | 自定义分组 scale | DeepSeek-V4 routed expert | 依赖实现 |
| INT4 | per-channel / per-group，有零点 | 社区量化权重 | 通用 |

共同点：**block/micro scaling**——每个小块用自己的 scale，
让块内 4 bit 用满动态范围。这是它们比朴素 INT4 精度高的根本原因。

本模块给出 NVFP4 与 MXFP4 的量化/反量化（**模拟**，无专用指令），
以及一个统一的 "按块选 scale" 抽象，方便对比三种格式的误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

__all__ = ["MicroScalingConfig", "nvfp4_quantize", "mxfp4_quantize",
           "micro_dequantize", "compare_formats", "FORMATS"]


FORMATS = ("nvfp4", "mxfp4", "int4")

FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@dataclass
class MicroScalingConfig:
    fmt: str = "nvfp4"
    block: int = 16            # nvfp4 用 16，mxfp4 用 32
    scale_bits: int = 8        # nvfp4 的 scale 本身是 E4M3


def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """把张量舍入到 FP4 的可表示值集合（E2M1）。"""
    flat = x.reshape(-1)
    vals = FP4_VALUES.to(flat.device, flat.dtype)
    # 最近邻：逐元素找距离最小的可表示值
    idx = (flat.unsqueeze(-1) - vals.unsqueeze(0)).abs().argmin(dim=-1)
    return vals[idx].reshape(x.shape)


def nvfp4_quantize(w: torch.Tensor, cfg: Optional[MicroScalingConfig] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4：每 16 个元素一个 E4M3 scale + FP4 码字。"""
    cfg = cfg or MicroScalingConfig(fmt="nvfp4", block=16)
    return _block_quantize(w, cfg.block, power_of_two=False)


def mxfp4_quantize(w: torch.Tensor, cfg: Optional[MicroScalingConfig] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP4：每 32 个元素共享一个 **2 的幂** scale（只需存指数，更省）。"""
    cfg = cfg or MicroScalingConfig(fmt="mxfp4", block=32)
    return _block_quantize(w, cfg.block, power_of_two=True)


def _block_quantize(w: torch.Tensor, block: int, power_of_two: bool
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = w.shape
    pad_n = (-N) % block
    x = torch.nn.functional.pad(w, (0, pad_n)) if pad_n else w
    nb = x.shape[1] // block
    xb = x.reshape(M, nb, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    if power_of_two:
        # 2 的幂 scale：只需存指数（log2），硬件上更省
        scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))       # FP4 最大值 6
    else:
        scale = amax / 6.0
    q = _round_to_fp4(xb / scale)
    return q.reshape(M, nb * block)[:, :N].reshape(-1), scale.reshape(M, nb)


def micro_dequantize(q: torch.Tensor, scale: torch.Tensor,
                     block: int, shape: Tuple[int, int]) -> torch.Tensor:
    M, N = shape
    nb = (N + block - 1) // block
    qb = q.reshape(M, nb, block)
    x = qb * scale.unsqueeze(-1)
    return x.reshape(M, nb * block)[:, :N]


def compare_formats(w: torch.Tensor) -> Dict[str, float]:
    """同一个权重在三种格式下的相对误差——这是选型时唯一可信的依据。"""
    out: Dict[str, float] = {}
    ref = w.abs().mean().clamp(min=1e-12)

    for fmt, block, pot in (("nvfp4", 16, False), ("mxfp4", 32, True)):
        q, s = _block_quantize(w, block, pot)
        d = micro_dequantize(q, s, block, w.shape)
        out[fmt] = float((d - w).abs().mean() / ref)

    # INT4 per-group 对照（对称、有零点偏移的 min/max 量化）
    b = 32
    pad = (-w.shape[1]) % b
    x = torch.nn.functional.pad(w, (0, pad)) if pad else w
    xb = x.reshape(w.shape[0], -1, b)
    lo, hi = xb.amin(-1, keepdim=True), xb.amax(-1, keepdim=True)
    sc = (hi - lo).clamp(min=1e-12) / 15.0
    qi = torch.round((xb - lo) / sc).clamp(0, 15)
    di = (qi * sc + lo).reshape(x.shape)[:, :w.shape[1]]
    out["int4"] = float((di - w).abs().mean() / ref)
    return out
