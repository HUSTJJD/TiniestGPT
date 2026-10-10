"""DPO 修补族 + 在线 DPO + 拒绝采样。

DPO 之后两年冒出一大堆"对 DPO 的修补"，各自针对一种失败模式：

| 方法 | 解决的问题 | 核心改动 |
|---|---|---|
| **IPO** | DPO 在确定性偏好下过拟合 | 把 log-sigmoid 换成平方损失 |
| **KTO** | 需要成对数据（贵） | 只要"好/坏"单条标注就能训 |
| **ORPO** | 需要参考模型（占显存） | 把 odds-ratio 项直接加进 SFT 损失 |
| **SimPO** | DPO 的长度偏差 + 仍需参考模型 | 用长度归一化的隐式奖励，去掉参考模型 |

**在线 DPO**（2026 取代离线 DPO）：每训一轮就用**当前模型**重新采样偏好对再训。
离线 DPO 的问题是分布漂移——参考数据来自旧模型，训着训着就不匹配了。

**拒绝采样 / best-of-n**：对可验证任务，采样 n 个、只保留通过的那些做 SFT。
这是"最朴素但极其有效"的 RL 替代品，Llama 3.1 405B 做了 5 轮。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["ipo_loss", "kto_loss", "orpo_loss", "simpo_loss",
           "OnlineDPOSampler", "rejection_sample", "best_of_n"]


def _logps_from(logits: torch.Tensor, labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        tok = tok * mask
    return tok.sum(dim=-1)


def ipo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """IPO：把 log-sigmoid 换成平方损失，缓解确定性偏好下的过拟合。"""
    y = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return ((y - 1.0 / (2 * beta)) ** 2).mean()


def kto_loss(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
             desirable: torch.Tensor, beta: float = 0.1,
             lam: float = 1.0) -> torch.Tensor:
    """KTO：只要"这条好不好"，不需要成对数据。

    ``desirable`` 是 bool 张量（True=可接受）。
    不可接受的样本用 **反向**的 log-sigmoid，这是 KTO 的关键设计。
    """
    kl = (policy_logp - ref_logp)
    z = beta * kl
    desirable_f = desirable.float()
    loss_pos = 1.0 - F.sigmoid(z - lam)
    loss_neg = 1.0 - F.sigmoid(lam - z)
    loss = desirable_f * loss_pos + (1.0 - desirable_f) * loss_neg
    return loss.mean()


def orpo_loss(nll: torch.Tensor, policy_chosen: torch.Tensor,
              policy_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """ORPO：SFT 损失 + odds-ratio 项，**不需要参考模型**。

    ``odds = p/(1-p)``，用 log-odds 差构造偏好项。
    """
    odds = lambda lp: lp - torch.log1p(-torch.exp(lp.clamp(max=-1e-7)))  # noqa: E731
    log_odds_diff = odds(policy_chosen) - odds(policy_rejected)
    pref = -F.logsigmoid(beta * log_odds_diff)
    return (nll + pref).mean()


def simpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
               len_chosen: torch.Tensor, len_rejected: torch.Tensor,
               beta: float = 2.0, gamma_beta_ratio: float = 0.1) -> torch.Tensor:
    """SimPO：长度归一化的隐式奖励，**去掉参考模型**，顺带治长度偏差。

    长度归一化是重点：DPO 的隐式奖励与序列长度正相关，
    于是模型学会"写更长"来刷分——这就是长度偏差。
    """
    r_c = policy_chosen / len_chosen.clamp(min=1)
    r_r = policy_rejected / len_rejected.clamp(min=1)
    gamma = beta * gamma_beta_ratio
    return -F.logsigmoid(beta * (r_c - r_r) - gamma).mean()


# --------------------------------------------------------------------------- #
@dataclass
class OnlineDPOSampler:
    """在线 DPO：每轮用**当前模型**重新采样，再打分构造偏好对。

    :param judge: ``(prompt, completion) -> float`` 的打分函数
    """

    judge: Callable[[str, str], float]
    group_size: int = 4
    pairs_per_round: int = 8

    def build_pairs(self, prompts: Sequence[str],
                    sampler: Callable[[str], List[str]]) -> List[Tuple[str, str, str]]:
        """返回 ``[(prompt, chosen, rejected), ...]``。"""
        pairs: List[Tuple[str, str, str]] = []
        for p in prompts:
            cands = sampler(p)
            if len(cands) < 2:
                continue
            scored = sorted(cands, key=lambda c: self.judge(p, c), reverse=True)
            pairs.append((p, scored[0], scored[-1]))
            if len(pairs) >= self.pairs_per_round:
                break
        return pairs

    def report(self) -> str:
        return f"OnlineDPO: group={self.group_size}, 每轮 {self.pairs_per_round} 对"


def rejection_sample(prompts: Sequence[str], sampler: Callable[[str], List[str]],
                     verifier: Callable[[str, str], bool],
                     keep_per_prompt: int = 1) -> List[Tuple[str, str]]:
    """拒绝采样：只保留**通过验证**的样本（用于迭代 SFT）。

    这是"最朴素但极其有效"的 RL 替代品——不需要 critic，不需要偏好对。
    """
    kept: List[Tuple[str, str]] = []
    for p in prompts:
        n = 0
        for c in sampler(p):
            if verifier(p, c):
                kept.append((p, c))
                n += 1
                if n >= keep_per_prompt:
                    break
    return kept


def best_of_n(prompt: str, sampler: Callable[[str], List[str]],
              scorer: Callable[[str, str], float], n: int = 8) -> Tuple[str, float]:
    """best-of-n：采样 n 个取最高分。推理时的"用算力换质量"。"""
    cands = sampler(prompt)[:n]
    if not cands:
        return "", 0.0
    scored = [(c, scorer(prompt, c)) for c in cands]
    return max(scored, key=lambda x: x[1])
