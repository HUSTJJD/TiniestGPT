"""FP8 训练（DeepSeek-V3/V4 路线）。

FP8 训练**不是**"把 bf16 换成 fp8"。它有三个必须同时成立的组件：

1. **分块缩放（block-wise scaling）**
   E4M3 的动态范围极窄（~±448）。逐张量缩放会被个别大值拖垮，
   所以按 ``block``（常用 128×128）给每个块一个 scale，
   让每个块内部都用满自己的动态范围。

2. **高精度主权重 + 高精度累加**
   前向/反向的 GEMM 走 FP8，但 **master weight 保持 fp32/bf16**，
   optimizer 状态也是高精度；累加用 fp32（Tensor Core 的 accumulate 本来就是）。
   低精度只用来省**带宽和算力**，不用来存"真相"。

3. **随机舍入（可选）**
   小更新量在 fp8 下会被系统性抹掉。随机舍入让舍入误差无偏，
   代价是要维护 RNG 状态。

本模块给的是**教学可用的最小实现**：
用 PyTorch 的 ``torch.float8_e4m3fn`` 做量化-反量化模拟（simulated FP8）——
在 sm_86 上没有真正的 FP8 Tensor Core，真实加速要在 Hopper 之后才有。
代码路径与真实实现一致，可以先把数值行为验证清楚。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["FP8Config", "quantize_blockwise", "FP8Linear", "FP8Manager",
           "has_fp8_support", "fp8_amax_history"]


FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0 if FP8_E4M3 is not None else 0.0


def has_fp8_support() -> bool:
    """是否**真的**有 FP8 dtype（cu118+ / Hopper+ 才有硬件加速）。"""
    return FP8_E4M3 is not None


@dataclass
class FP8Config:
    block: int = 128              # 分块缩放的块边长
    stochastic_rounding: bool = False
    amax_history: int = 64        # 用历史 amax 的窗口估计 scale（防单步尖刺）
    keep_high_precision: Tuple[str, ...] = (
        "norm", "embed", "lm_head", "gate", "bias",
    )   # 这些模块参数少但数值敏感，保持高精度


def quantize_blockwise(x: torch.Tensor, block: int, stochastic: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 ``block×block`` 分块量化到 E4M3。返回 ``(q_fp8, scales)``。

    ``scales`` 的形状是 ``[ceil(M/block), ceil(N/block)]``；
    反量化时按块 broadcast 乘回去。
    """
    if FP8_E4M3 is None:
        raise RuntimeError("当前 torch 不支持 float8_e4m3fn")
    M, N = x.shape
    bm, bn = (M + block - 1) // block, (N + block - 1) // block
    pad_m, pad_n = bm * block - M, bn * block - N
    if pad_m or pad_n:
        x = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m))
    xb = x.reshape(bm, block, bn, block).permute(0, 2, 1, 3)     # [bm,bn,block,block]
    amax = xb.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-12)
    scale = FP8_MAX / amax
    q = (xb * scale).clamp(-FP8_MAX, FP8_MAX)
    if stochastic:
        # 随机舍入：向下取整 + 以小数部分为概率向上取整，使舍入误差无偏
        floor = torch.floor(q)
        frac = q - floor
        q = floor + (torch.rand_like(frac) < frac).to(q.dtype)
    q = q.to(FP8_E4M3)
    # amax/scale 的形状是 [bm,bn,1,1]，去掉两个尾部维度得到 [bm,bn]
    return q.permute(0, 2, 1, 3).reshape(bm * block, bn * block), \
        scale.squeeze(-1).squeeze(-1)


def dequantize_blockwise(q: torch.Tensor, scales: torch.Tensor,
                         block: int, shape: Tuple[int, int]) -> torch.Tensor:
    M, N = shape
    bm, bn = scales.shape
    # 与量化时**完全对称**的排布：先 reshape 成 [bm,block,bn,block] 再 permute
    xb = q.reshape(bm, block, bn, block).permute(0, 2, 1, 3)     # [bm,bn,block,block]
    x = xb.float() / scales.unsqueeze(-1).unsqueeze(-1)
    return x.permute(0, 2, 1, 3).reshape(bm * block, bn * block)[:M, :N]


def fp8_amax_history(hist: List[float], amax: float, window: int) -> float:
    """用滑动窗口内的**最大** amax 估计 scale：单步尖刺不至于把 scale 拉爆。"""
    hist.append(float(amax))
    if len(hist) > window:
        hist.pop(0)
    return max(hist)


class FP8Linear(nn.Module):
    """在 forward 时把权重临时量化成 FP8 做 matmul（主权重仍是高精度）。"""

    def __init__(self, linear: nn.Linear, cfg: FP8Config) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.cfg = cfg
        self._amax_hist: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if FP8_E4M3 is None or self.cfg.block <= 0:
            return torch.nn.functional.linear(x, w, self.bias)
        q, scales = quantize_blockwise(w.detach(), self.cfg.block,
                                       self.cfg.stochastic_rounding)
        wd = dequantize_blockwise(q, scales, self.cfg.block, w.shape).to(w.dtype)
        return torch.nn.functional.linear(x, wd, self.bias)


class FP8Manager:
    """把模型里的 Linear 换成 FP8Linear（**保持主权重对象不变**，所以优化器不用改）。"""

    def __init__(self, cfg: Optional[FP8Config] = None) -> None:
        self.cfg = cfg or FP8Config()
        self.replaced: Dict[str, nn.Module] = {}

    def _keep_high(self, name: str) -> bool:
        low = name.lower()
        return any(k in low for k in self.cfg.keep_high_precision)

    def apply(self, model: nn.Module) -> int:
        n = 0
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if self._keep_high(name):
                continue
            parent, child = _parent_of(model, name)
            if parent is None:
                continue
            setattr(parent, child, FP8Linear(mod, self.cfg))
            self.replaced[name] = mod
            n += 1
        return n

    def report(self) -> str:
        if FP8_E4M3 is None:
            return "FP8: 当前环境无 float8 支持，已回退到高精度 matmul"
        return f"FP8: {len(self.replaced)} 个 Linear 走分块 E4M3（block={self.cfg.block}）"


def _parent_of(model: nn.Module, qualified: str):
    parts = qualified.split(".")
    if not parts:
        return None, None
    cur = model
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return None, None
        cur = getattr(cur, p)
    return cur, parts[-1]
