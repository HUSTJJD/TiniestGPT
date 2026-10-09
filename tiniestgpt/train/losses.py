"""损失函数。

* **交叉熵**：自回归语言模型的唯一主损失。
* **Label smoothing**：把 one-hot 软化，防止模型过度自信（对校准有帮助，
  但会略微损失 log-likelihood，预训练一般不开，SFT 时可小量使用）。
* **z-loss**（ST-MoE / PaLM）：``λ · mean(logsumexp(logits)²)``。
  它惩罚"logits 整体的平移"，能显著减少训练中的 loss 尖刺与
  bf16 下的数值不稳定，是"大模型训练必备的小技巧"。
* **MoE 辅助损失**：见 ``model/moe.py``，这里负责加权合并。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = ["language_modeling_loss", "z_loss", "perplexity", "mtp_loss"]


def mtp_loss(mtp_logits: list, labels: torch.Tensor, weight: float = 0.1,
             ignore_index: int = -100) -> Dict[str, torch.Tensor]:
    """MTP（多 token 预测）的辅助损失。

    ``labels[t] = x_{t+1}``，因此第 k 个 MTP 头（k 从 0 开始，预测 ``x_{t+2+k}``）
    在位置 t 的目标是 ``labels[t + k + 1]``::

        loss_k = CE(mtp_logits[k][:, :-k-1], labels[:, k+1:])

    它**不影响推理时的主 logits**——MTP 头只在训练中提供额外的学习信号，
    推理时它们改行当推测解码的草稿器。
    """
    if not mtp_logits or weight <= 0:
        return {"mtp": labels.new_zeros(()) if hasattr(labels, "new_zeros")
                else torch.zeros(())}
    total = None
    for k, lg in enumerate(mtp_logits):
        T = lg.shape[1]
        if T <= k + 1:
            continue
        ce = F.cross_entropy(lg[:, :-(k + 1)].float().reshape(-1, lg.shape[-1]),
                             labels[:, k + 1:].reshape(-1), ignore_index=ignore_index)
        total = ce if total is None else total + ce
    if total is None:
        return {"mtp": torch.zeros((), device=labels.device)}
    return {"mtp": weight * total}


def z_loss(logits: torch.Tensor, weight: float = 1e-4) -> torch.Tensor:
    """logits 的 log-sum-exp 平方均值（对平移敏感，对分布形状不敏感）。"""
    lse = torch.logsumexp(logits.float(), dim=-1)
    return weight * (lse ** 2).mean()


def perplexity(loss: float) -> float:
    import math

    return math.exp(min(loss, 20.0))


def language_modeling_loss(
    logits: torch.Tensor,                 # [B, T, V]
    labels: torch.Tensor,                 # [B, T]
    label_smoothing: float = 0.0,
    z_loss_weight: float = 0.0,
    aux_loss: Optional[torch.Tensor] = None,
    aux_weight: float = 0.0,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """返回 ``{"loss": 总损失, "ce": ..., "z": ..., "aux": ...}``。"""
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.float().reshape(-1, V),
        labels.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    total = ce
    z = logits.new_zeros(())
    if z_loss_weight > 0:
        mask = (labels != ignore_index)
        if mask.any():
            z = z_loss(logits.reshape(-1, V)[mask.reshape(-1)], z_loss_weight)
            total = total + z
    aux = logits.new_zeros(())
    if aux_loss is not None and aux_weight > 0:
        aux = aux_weight * aux_loss
        total = total + aux
    return {"loss": total, "ce": ce.detach(), "z": z.detach(), "aux": aux.detach()}
