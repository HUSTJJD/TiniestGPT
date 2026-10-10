"""PPO + Critic + GAE：GRPO 之前的主流，也是理解"为什么 GRPO 更省"的参照物。

PPO 要同时维护 **4 个模型**：

    actor（被训练的政策） / critic（价值网络） / reference（KL 约束） / reward（打分）

显存是 SFT 的 4 倍以上，这也是 RLHF 工程复杂度的根源。
GRPO 用"组内平均奖励"替代 critic，直接砍掉一个模型——
但要真正体会这一点，得先把 PPO 跑起来。

本模块提供：

* :func:`compute_gae` —— 广义优势估计（λ 控制偏差-方差权衡）；
* :class:`Critic` —— 价值网络（从 hidden state 回归标量）；
* :func:`ppo_loss` —— clipped surrogate + value loss + entropy bonus + KL。

**KL 的估计**：常用 k3（Schulman 的无偏估计），比朴素 KL 方差小得多。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PPOConfig", "Critic", "compute_gae", "compute_returns",
           "ppo_loss", "value_loss", "entropy_from_logits", "kl_penalty"]


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95          # GAE 的 λ
    kl_kind: str = "k3"


class Critic(nn.Module):
    """价值网络：把 hidden state 池化后回归一个标量。"""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)          # [..., ]


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float = 1.0, lam: float = 0.95
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计。

    ``rewards/values`` 形状 ``[T]`` 或 ``[B,T]``，``dones`` 同形状（1=该步后终止）。
    返回 ``(advantages, returns)``。
    """
    T = rewards.shape[-1]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[..., :1]).squeeze(-1) if rewards.dim() > 1 \
        else torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(gae)
    for t in range(T - 1, -1, -1):
        r = rewards[..., t] if rewards.dim() > 1 else rewards[t]
        v = values[..., t] if values.dim() > 1 else values[t]
        nv = values[..., t + 1] if t + 1 < T else torch.zeros_like(v)
        done = dones[..., t] if dones.dim() > 1 else dones[t]
        delta = r + gamma * nv * (1.0 - done) - v
        gae = delta + gamma * lam * (1.0 - done) * gae
        if adv.dim() > 1:
            adv[..., t] = gae
        else:
            adv[t] = gae
    returns = adv + values
    return adv, returns


def compute_returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """折扣回报（GAE 的退化版本，λ=1 时等价）。"""
    T = rewards.shape[-1]
    out = torch.zeros_like(rewards)
    acc = torch.zeros_like(rewards[..., 0] if rewards.dim() > 1 else rewards[0])
    for t in range(T - 1, -1, -1):
        r = rewards[..., t] if rewards.dim() > 1 else rewards[t]
        acc = r + gamma * acc
        if out.dim() > 1:
            out[..., t] = acc
        else:
            out[t] = acc
    return out


def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor, kind: str = "k3"
               ) -> torch.Tensor:
    """三种常见 KL 估计。

    * ``k1``：朴素 ``logp - ref_logp``，有偏但简单；
    * ``k2``：``0.5 (log-ratio)²``，近似 KL，恒正；
    * ``k3``：Schulman 的 ``exp(r) - r - 1``，无偏且方差更小——默认用它。
    """
    r = logp - ref_logp
    if kind == "k1":
        return r
    if kind == "k2":
        return 0.5 * r.pow(2)
    return torch.exp(r) - r - 1.0


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), dim=-1)
    p = lp.exp()
    return -(p * lp).sum(dim=-1)


def value_loss(values: torch.Tensor, returns: torch.Tensor,
               old_values: Optional[torch.Tensor] = None,
               clip: float = 0.2) -> torch.Tensor:
    """价值函数的 clipped 损失（与 policy 一样的"别走太远"思路）。"""
    if old_values is not None and clip > 0:
        v_clip = old_values + (values - old_values).clamp(-clip, clip)
        return 0.5 * torch.maximum((values - returns).pow(2),
                                   (v_clip - returns).pow(2)).mean()
    return 0.5 * (values - returns).pow(2).mean()


def ppo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor,
             advantages: torch.Tensor, cfg: PPOConfig,
             ref_logp: Optional[torch.Tensor] = None,
             entropy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """PPO 的 clipped surrogate 目标。

    :param new_logp/old_logp: ``[B,T]`` 的逐 token log 概率
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = advantages
    if adv.dim() == 1 and ratio.dim() == 2:
        adv = adv.unsqueeze(1)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy = -torch.minimum(unclipped, clipped).mean()

    total = policy
    kl = new_logp.new_zeros(())
    if ref_logp is not None and cfg.kl_coef > 0:
        kl = kl_penalty(new_logp, ref_logp, cfg.kl_kind).mean()
        total = total + cfg.kl_coef * kl
    ent = new_logp.new_zeros(())
    if entropy is not None and cfg.entropy_coef > 0:
        ent = entropy.mean()
        total = total - cfg.entropy_coef * ent      # 熵越大越好 → 减
    return {"loss": total, "policy": policy.detach(), "kl": kl.detach(),
            "entropy": ent.detach(),
            "clip_frac": (ratio - 1.0).abs().gt(cfg.clip_eps).float().mean().detach()}
