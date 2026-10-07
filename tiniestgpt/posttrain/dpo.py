"""DPO / IPO：直接从偏好数据学习，不需要奖励模型。

核心恒等式（RLHF 的漂亮结论）：
最优策略与奖励的关系是 ``r(x,y) = β·log(π(y|x) / π_ref(y|x)) + const``，
于是偏好概率可写成只含策略的形式，从而**绕开显式奖励模型**：

    L_DPO = -log σ( β·[(logπ(y_w) - logπ_ref(y_w)) - (logπ(y_l) - logπ_ref(y_l))] )

工程要点：
  * 必须先算一份 **reference logps**（参考模型前向，no_grad）并缓存，
    否则每步要跑两次模型；
  * ``β`` 越大越保守（越贴近参考模型）；典型 0.1；
  * IPO 把 sigmoid 换成平方损失，能缓解 DPO 在弱偏好对上的过拟合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = ["DPOConfig", "dpo_loss", "ipo_loss", "sequence_logps"]


@dataclass
class DPOConfig:
    beta: float = 0.1
    loss_type: str = "dpo"        # dpo | ipo | hinge
    label_smoothing: float = 0.0
    max_len: int = 512


@torch.no_grad()
def sequence_logps(model: torch.nn.Module, input_ids: torch.Tensor,
                   labels: torch.Tensor, attn_mask=None) -> torch.Tensor:
    """计算每条序列的 **token 平均** 对数概率（DPO 用 sum 更常见，这里返回 sum）。"""
    logits = model(input_ids, attn_mask=attn_mask)
    logp = torch.log_softmax(logits.float(), dim=-1)
    mask = labels != -100
    tok_logp = torch.gather(logp, 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    return tok_logp.sum(-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1, loss_type: str = "dpo",
             label_smoothing: float = 0.0) -> Dict[str, torch.Tensor]:
    """四种偏好损失的统一实现。"""
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    if loss_type == "ipo":
        loss = F.mse_loss(logits / beta, torch.ones_like(logits) / (2 * beta))
    elif loss_type == "hinge":
        loss = F.relu(1 - logits).mean()
    else:
        if label_smoothing > 0:
            loss = (-F.logsigmoid(logits) * (1 - label_smoothing)
                    - F.logsigmoid(-logits) * label_smoothing).mean()
        else:
            loss = -F.logsigmoid(logits).mean()
    # 诊断指标：隐含奖励的准确率（>0 即判对）
    acc = (logits > 0).float().mean()
    margin = logits.mean().detach()
    return {"loss": loss, "accuracy": acc.detach(), "margin": margin}
