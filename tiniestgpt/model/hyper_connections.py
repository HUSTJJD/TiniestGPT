"""mHC：Manifold-Constrained Hyper-Connections（DeepSeek-V4）。

标准 Pre-Norm Transformer 里所有子层都往**同一条**残差流上累加：

    x' = x + Attention(Norm(x))
    x'' = x' + FFN(Norm(x'))

深层大 MoE 中这条流上的表示可能在部分方向被不断放大。
mHC 维护 ``hc_mult`` 条残差流（V4 用 4 条）：

    X ∈ R^[m, D]
    pre-mix :  x_in = combine(X)             # 从 m 条流读出子层输入
    sub-layer: y = F(x_in)
    post-mix:  X' = P @ X + inject(y)        # P 是 m×m 混合矩阵

**关键在约束**：P 通过 Sinkhorn-Knopp 迭代投影成近似**双随机矩阵**
（行列和都为 1、元素非负）。双随机约束把残差混合的谱范数限制住，
防止无界放大，同时仍允许模型在 m 条流之间重排信息。

系统代价不是参数量，而是 **activation**：
    普通残差: [B, L, D]
    mHC(4):   [B, L, 4, D]
所以训练框架必须用融合与重计算控制显存，不能直接把所有中间流都存下来。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["sinkhorn_knopp", "HyperConnections", "MHCBlock"]


def sinkhorn_knopp(logits: torch.Tensor, iters: int = 20,
                   eps: float = 1e-8) -> torch.Tensor:
    """把任意实矩阵投影成近似双随机矩阵（行和=1，列和=1，元素≥0）。

    交替做行归一化 / 列归一化即可；这是 Sinkhorn-Knopp 的标准形式。
    """
    x = logits.float()
    for _ in range(iters):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)      # 行方向
        x = x - torch.logsumexp(x, dim=-2, keepdim=True)      # 列方向
    return torch.exp(x).clamp_min(eps)


class HyperConnections(nn.Module):
    """m 条残差流的 pre-mix / post-mix。"""

    def __init__(self, dim: int, mult: int = 4, iters: int = 20,
                 init_width: float = 0.02) -> None:
        super().__init__()
        self.dim, self.mult, self.iters = dim, mult, iters
        # 宽度方向的读出/写入与深度方向的混合，分开参数化
        self.read = nn.Parameter(torch.randn(mult) * init_width + 1.0 / mult)
        self.write = nn.Parameter(torch.randn(mult) * init_width + 1.0 / mult)
        self.mix_logits = nn.Parameter(torch.zeros(mult, mult))
        self._P: Optional[torch.Tensor] = None

    def mixing_matrix(self) -> torch.Tensor:
        if self._P is None or self.training:
            self._P = sinkhorn_knopp(self.mix_logits, self.iters)
        return self._P

    def width_init(self, batch: int, device, dtype) -> torch.Tensor:
        """初始化 m 条流：把它们都置成同一份输入（否则训练起步不稳）。"""
        raise NotImplementedError

    def pre_mix(self, X: torch.Tensor) -> torch.Tensor:
        """X:[B,L,m,D] → x_in:[B,L,D]"""
        return torch.einsum("blmd,m->bld", X, self.read)

    def post_mix(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """X:[B,L,m,D], y:[B,L,D] → X':[B,L,m,D]"""
        P = self.mixing_matrix().to(X.dtype)
        mixed = torch.einsum("mn,blnd->blmd", P, X)
        return mixed + self.write.view(1, 1, self.mult, 1) * y.unsqueeze(2)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mult={self.mult}, sinkhorn_iters={self.iters}"


class MHCBlock(nn.Module):
    """把任意子层（Attention / FFN / MoE）包进 mHC 残差。

    用法::

        hc = HyperConnections(dim, mult=4)
        blk = MHCBlock(hc, nn.Sequential(norm, attn))
        X = blk(X)          # X 的形状始终是 [B, L, m, D]
    """

    def __init__(self, hc: HyperConnections, sublayer: nn.Module) -> None:
        super().__init__()
        self.hc = hc
        self.sublayer = sublayer

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_in = self.hc.pre_mix(X)
        y = self.sublayer(x_in)
        return self.hc.post_mix(X, y)

    @staticmethod
    def expand(x: torch.Tensor, mult: int) -> torch.Tensor:
        """把普通残差 [B,L,D] 扩成 m 条流（全部初始化为同一份）。"""
        return x.unsqueeze(2).expand(*x.shape[:2], mult, x.shape[-1]).contiguous()

    @staticmethod
    def collapse(X: torch.Tensor) -> torch.Tensor:
        """把 m 条流并回一条：取均值（等价于 read=1/m 的均匀读出）。"""
        return X.mean(dim=2)
