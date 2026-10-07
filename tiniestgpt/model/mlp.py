"""前馈网络（FFN）：门控激活的现代形态。

* **SwiGLU**（LLaMA/Qwen/Mistral 标配）：``SiLU(xW_gate) ⊙ (xW_up)``，
  相比 ReLU/GELU MLP 在同等参数量下有更低的 loss，代价是多一次矩阵乘。
* **GEGLU / ReGLU**：把门控换成 GELU / ReLU²。
* 门控与上投影**融合成一次 GEMM**（``w_in: dim → 2·hidden``）能省一次 kernel launch，
  是推理引擎里最常见的"免费"优化之一。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeedForward", "get_activation"]


def get_activation(name: str):
    name = name.lower()
    if name in ("silu", "swish"):
        return F.silu
    if name == "gelu":
        return lambda x: F.gelu(x, approximate="tanh")
    if name in ("relu2", "squared_relu"):
        return lambda x: F.relu(x).pow(2)
    if name == "relu":
        return F.relu
    raise ValueError(f"未知激活函数: {name}")


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act_type: str = "swiglu",
                 dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.act_type = act_type.lower()
        self.gated = self.act_type in ("swiglu", "geglu", "reglu", "reglu2")
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.gated:
            self.w_in = nn.Linear(dim, 2 * hidden_dim, bias=bias)   # 融合 gate/up
        else:
            self.w_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)

        act_map = {"swiglu": "silu", "geglu": "gelu", "reglu": "relu", "reglu2": "relu2"}
        self.act = get_activation(act_map.get(self.act_type, self.act_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_in(x)
        if self.gated:
            gate, up = h.chunk(2, dim=-1)
            h = self.act(gate) * up
        else:
            h = self.act(h)
        return self.w_out(self.dropout(h))
