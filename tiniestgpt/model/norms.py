"""归一化层。

* **RMSNorm**（LLaMA/Qwen/DeepSeek 标配）：只除以均方根，不减均值。
  比 LayerNorm 少一次减均值与 bias，训练更稳且更快。
* **Sandwich / Post-Norm**：在残差分支之后再加一道 norm（Qwen3、Gemma2）。
  能显著抑制深层 activation 的"数值爆炸"，代价是少量额外计算。
* **QK-Norm**：对 Q/K 向量本身做归一化（而非整个 hidden），
  让 attention logits 的量级与层数解耦，是长上下文与低精度训练的稳定器。
* **Dynamic Tanh (DyT)**：2025 年的新观点——用 ``tanh(α·x)`` 替代归一化，
  在不少场景下能打平 RMSNorm 且更快（对归一化"重新中心化"假说的直接挑战）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "LayerNorm", "DynamicTanh", "build_norm", "QKNorm"]


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x²) + eps) * (1 + w)"""

    def __init__(self, dim: int, eps: float = 1e-6, init_ones: bool = True,
                 elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim) if init_ones else torch.zeros(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一律用 float32 计算统计量，避免 bf16 下精度损失
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = xf * rms
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(dtype)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        mean = xf.mean(-1, keepdim=True)
        var = xf.var(-1, keepdim=True, unbiased=False)
        out = (xf - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight.float() + self.bias.float()
        return out.to(dtype)


class DynamicTanh(nn.Module):
    """DyT:  y = tanh(α · x) · w + b  （α 可学习，替代 RMSNorm）"""

    def __init__(self, dim: int, init_alpha: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), init_alpha))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = torch.tanh(self.alpha.float() * x.float())
        return (y * self.weight.float() + self.bias.float()).to(dtype)


class QKNorm(nn.Module):
    """对 head 维度做 RMSNorm（不带可学习参数），用于 Q/K。"""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype)


def build_norm(name: str, dim: int, eps: float = 1e-6, init_ones: bool = True) -> nn.Module:
    name = name.lower()
    if name == "rms":
        return RMSNorm(dim, eps=eps, init_ones=init_ones)
    if name in ("layernorm", "ln"):
        return LayerNorm(dim, eps=eps)
    if name in ("dynamic_tanh", "dyt"):
        return DynamicTanh(dim)
    raise ValueError(f"未知归一化类型: {name}")
