"""MTP（Multi-Token Prediction，多 token 预测）：2026 年旗舰模型的标配训练目标。

普通语言模型只预测下一个 token：

    p(x_{t+1} | x_{≤t})

MTP 同时监督未来 n 个位置::

    L_MTP = Σ_k λ_k · CE(p_k(x_{t+k} | x_{≤t}), x_{t+k})

为什么 2026 年人人都在用（DeepSeek-V3/V4、Kimi、MiniMax、GLM-5.2、Nemotron）：

1. **训练侧**：多任务监督让隐藏状态"多想一步"，单位 token 的学习信号更密；
2. **推理侧**：MTP head 天然就是**推测解码的草稿器**——
   一次前向就能吐出 n 个候选 token，再由主模型**一次前向批量验证**。
   这比外挂一个小 draft 模型便宜得多（不用额外权重、不用维护两个 KV Cache）。

**但 MTP 不保证加速**，这点必须讲清楚。收益取决于：

* 平均接受长度 τ；
* draft head 自身的额外计算；
* batch 大时验证阶段本来就很划算 → 投机反而可能亏；
* CUDA Graph 能否覆盖动态接受长度。

``benchmarks/inference_ablation.py`` 里加了 L6 档，用来把这件事量化出来。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .norms import build_norm

__all__ = ["MTPModule"]


class MTPModule(nn.Module):
    """n 个 MTP 预测头。

    第 k 个头从位置 t 的隐藏状态预测位置 ``t + 1 + k`` 的 token。
    头的结构刻意做小（一层 norm + 一层线性 + 共享 ``lm_head``），
    因为它的定位是"便宜的草稿器"，不是第二个模型。
    """

    def __init__(self, cfg: ModelConfig, n_predict: int = 1) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_predict = n_predict
        self.norms = nn.ModuleList([
            build_norm(cfg.norm_type, cfg.dim, cfg.norm_eps) for _ in range(n_predict)
        ])
        self.projs = nn.ModuleList([
            nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(n_predict)
        ])
        # 深度越远的头越不确定 → 初始化更小，避免训练早期污染主损失
        for k, p in enumerate(self.projs):
            nn.init.normal_(p.weight, std=cfg.init_std / (k + 1))

    def forward(self, hidden: torch.Tensor, lm_head: nn.Module) -> List[torch.Tensor]:
        """``hidden: [B,T,D]``（已过 norm_f）→ n 份 logits ``[B,T,V]``。

        :return: 第 k 份 logits 的第 t 个位置，是对 ``x_{t+1+k}`` 的预测。
        """
        return [lm_head(proj(norm(hidden))) for norm, proj in zip(self.norms, self.projs)]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
