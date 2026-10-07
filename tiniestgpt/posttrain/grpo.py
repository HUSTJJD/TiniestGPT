"""GRPO（Group Relative Policy Optimization）：DeepSeek-R1 同族的 RL 算法。

与 PPO 的区别：**没有 critic**。
对同一个 prompt 采样 G 条回答 → 用奖励函数打分 → 在**组内**做标准化：

    Â_i = (r_i - mean(r)) / (std(r) + eps)

这一步"组内相对比较"就是优势估计，从而省掉一个与策略同规模的价值网络，
显存直接减半，且在小模型上非常稳定。

策略更新是 PPO 的 clipped surrogate（防止单步更新过猛），
外加一个对参考模型的 KL 惩罚（可选用无偏的 k3 估计器降低方差）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = ["GRPOConfig", "compute_group_advantages", "grpo_loss", "kl_penalty"]


@dataclass
class GRPOConfig:
    group_size: int = 8          # 每个 prompt 采样多少条回答
    beta: float = 0.04           # KL 惩罚系数（0 表示不约束）
    eps_clip: float = 0.2        # PPO clip 范围
    clip_higher: float = 0.28    # DAPO 的 clip-higher：放宽上界，鼓励探索
    kl_estimator: str = "k3"     # k1(有偏,低方差) | k3(无偏)
    loss_type: str = "token_mean"  # token_mean | seq_mean（DAPO 用后者，避免长序列主导）
    max_len: int = 256
    temperature: float = 1.0
    std_eps: float = 1e-4


def compute_group_advantages(rewards: torch.Tensor, group_size: int,
                             std_eps: float = 1e-4) -> torch.Tensor:
    """组内标准化：``(r - mean) / (std + eps)``。

    :param rewards: [P * G] 按 prompt 分组（连续 group_size 个为一组）
    """
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    adv = (r - mean) / (std + std_eps)
    return adv.view(-1)


def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor, estimator: str = "k3") -> torch.Tensor:
    """KL(π || π_ref) 的两种估计。

    * k1 = logp_ref - logp          （简单有偏，可能为负）
    * k3 = exp(ref - logp) - (ref - logp) - 1   （Schulman 的无偏估计，方差大但恒正）
    """
    d = ref_logp - logp
    if estimator == "k1":
        return d
    return torch.exp(d) - d - 1.0


def grpo_loss(
    logps: torch.Tensor,              # [N, T] 当前策略的 token logp
    old_logps: torch.Tensor,          # [N, T] 采样时策略的 token logp（用于重要性采样比）
    advantages: torch.Tensor,         # [N]  组内优势
    mask: torch.Tensor,               # [N, T] 有效 token 掩码
    ref_logps: Optional[torch.Tensor] = None,
    cfg: Optional[GRPOConfig] = None,
) -> Dict[str, torch.Tensor]:
    cfg = cfg or GRPOConfig()

    ratio = torch.exp(logps - old_logps)                      # [N, T]
    adv = advantages.unsqueeze(1).expand_as(ratio)

    # DAPO 的 clip-higher：上界比下界宽，鼓励低概率 token 被探索
    if cfg.clip_higher > cfg.eps_clip:
        pg = torch.min(ratio * adv,
                       torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.clip_higher) * adv)
    else:
        pg = torch.min(ratio * adv,
                       torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip) * adv)

    if cfg.loss_type == "seq_mean":
        # 先对每条序列取均值，再对 batch 取均值 → 避免长序列主导梯度
        loss = -((pg * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    else:
        loss = -(pg * mask).sum() / mask.sum().clamp(min=1)

    kl = logps.new_zeros(())
    if ref_logps is not None and cfg.beta > 0:
        k = kl_penalty(logps, ref_logps, cfg.kl_estimator)
        kl = (k * mask).sum() / mask.sum().clamp(min=1)
        loss = loss + cfg.beta * kl

    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio.clamp(min=1e-6)))
        clip_frac = (approx_kl.abs() > cfg.eps_clip).float().mean()
    return {"loss": loss, "kl": kl.detach(), "clip_frac": clip_frac,
            "ratio_mean": ratio[mask.bool()].mean().detach()}
