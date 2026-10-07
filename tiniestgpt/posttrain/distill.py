"""知识蒸馏：把"大教师"的暗知识（dark knowledge）压进小模型。

* **logit 蒸馏**：``KL(student/T || teacher/T) · T²``。
  温度 T 把教师的概率分布"摊平"，暴露出类别之间的相对关系
  （"猫 vs 狗"这种软标签比 one-hot 信息量大得多），T² 用于还原梯度量级。
* **隐层蒸馏**：用一层可学习的投影把学生隐层对齐到教师维度再做 MSE / cosine，
  对小模型尤其有效（学习"中间表示"比只学输出更容易）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["kd_loss", "hidden_state_loss", "DistillProjector"]


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0, alpha: float = 0.5,
            labels: Optional[torch.Tensor] = None) -> dict:
    """``alpha·CE(labels) + (1-alpha)·T²·KL(student||teacher)``"""
    T = temperature
    s = F.log_softmax(student_logits.float() / T, dim=-1)
    t = F.softmax(teacher_logits.float() / T, dim=-1)
    kd = F.kl_div(s, t, reduction="batchmean") * (T * T)
    out = {"kd": kd}
    if labels is not None:
        V = student_logits.shape[-1]
        ce = F.cross_entropy(student_logits.float().reshape(-1, V), labels.reshape(-1),
                             ignore_index=-100)
        out["ce"] = ce
        out["loss"] = alpha * ce + (1 - alpha) * kd
    else:
        out["loss"] = kd
    return out


def hidden_state_loss(student_h: torch.Tensor, teacher_h: torch.Tensor,
                      projector: Optional[nn.Module] = None,
                      kind: str = "mse") -> torch.Tensor:
    if projector is not None:
        student_h = projector(student_h)
    if student_h.shape != teacher_h.shape:
        raise ValueError(f"隐层维度不一致: {student_h.shape} vs {teacher_h.shape}")
    if kind == "cosine":
        return 1 - F.cosine_similarity(student_h, teacher_h.detach(), dim=-1).mean()
    return F.mse_loss(student_h, teacher_h.detach())


class DistillProjector(nn.Module):
    """把学生隐层投影到教师维度（训练时学习，推理时丢弃）。"""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
