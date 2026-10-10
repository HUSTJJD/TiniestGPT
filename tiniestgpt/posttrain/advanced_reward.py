"""进阶奖励工程：PRM / RLAIF / RLOO / 奖励塑形 / Thinking 预算。

**PRM（过程奖励模型）**
给每一步推理打分，而不是只给最终结果。OpenAI 的 "Let's Verify Step by Step"
在 MATH 上把解题率推到 78%。但工业落地一波三折：

* 通用任务的"正确步骤"难以定义（不像数学有标准解法）；
* PRM 本身需要大量人工标注，规模化困难；
* 容易被"看似合理但错误的中间步骤"骗到。

DeepSeek-R1 因此选择 outcome-based GRPO。**这是 2026 的事实标准，
但 PRM 仍是开放问题**——本模块给出实现，同时把这三点写在 docstring 里。

**RLAIF / 宪法反馈**
用 AI 生成的偏好标注替代人类标注。关键不是省钱，而是**可扩展**：
对齐阶段不再卡在标注员的吞吐量上。

**RLOO**
GRPO 用"组内均值"做 baseline；RLOO 用 **leave-one-out** 均值
（自己不参与自己的 baseline），偏差更小，也是常见的 GRPO 变体。

**奖励塑形**
长度、格式、重复都要显式塑形，否则模型会找到最省力的刷分路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ProcessRewardModel", "PRMConfig", "prm_loss", "aggregate_prm",
           "ConstitutionalJudge", "rloo_advantages", "shaped_reward",
           "RewardShaping", "ThinkingBudget"]


@dataclass
class PRMConfig:
    dim: int = 384
    hidden: int = 256
    sep_token: str = "\n\n"          # 步骤分隔符


class ProcessRewardModel(nn.Module):
    """过程奖励模型：对每个**步骤**打一个 0~1 的分数。"""

    def __init__(self, cfg: PRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scorer = nn.Sequential(
            nn.Linear(cfg.dim, cfg.hidden), nn.Tanh(),
            nn.Linear(cfg.hidden, 1))

    def forward(self, step_hidden: torch.Tensor) -> torch.Tensor:
        """``step_hidden``: [B, S, dim]（每个步骤末尾的隐藏状态）→ [B, S]"""
        return torch.sigmoid(self.scorer(step_hidden).squeeze(-1))


def prm_loss(pred: torch.Tensor, labels: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """PRM 用逐步骤的二分类交叉熵训练。"""
    loss = F.binary_cross_entropy(pred, labels.float(), reduction="none")
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def aggregate_prm(scores: torch.Tensor, mode: str = "min") -> torch.Tensor:
    """把逐步骤分数聚合成整条轨迹的分数。

    ``min`` 最严格（一步错整条错，数学题常用）；
    ``prod`` 连乘（长链条下会迅速趋 0）；
    ``last`` 只看最后一步（等价于 ORM）。
    """
    if mode == "min":
        return scores.min(dim=-1).values
    if mode == "prod":
        return scores.prod(dim=-1)
    if mode == "mean":
        return scores.mean(dim=-1)
    return scores[..., -1]


# --------------------------------------------------------------------------- #
@dataclass
class ConstitutionalJudge:
    """宪法式 AI 反馈（RLAIF）：用规则 + 模型自评生成偏好标注。

    :param principles: 自然语言规则列表，每条都能被追溯到一条"宪法"
    """

    principles: List[str] = field(default_factory=lambda: [
        "选择更有帮助的回答",
        "选择更安全的回答",
        "在不确定的时候选择承认不确定性的回答",
    ])
    calls: int = 0

    def judge(self, prompt: str, a: str, b: str,
              llm: Optional[Callable[[str], str]] = None) -> int:
        """返回 1（a 更好）/ -1（b 更好）/ 0（难分）。"""
        self.calls += 1
        if llm is None:
            # 无模型时退化为可解释的启发式：更长且更具体的通常更好
            sa = len(a) + 3 * a.count("\n")
            sb = len(b) + 3 * b.count("\n")
            return 1 if sa > sb else (-1 if sb > sa else 0)
        q = "宪法原则：\n" + "\n".join(f"- {p}" for p in self.principles)
        q += f"\n\n问题：{prompt}\n\nA：{a}\n\nB：{b}\n\n哪个更好？只回答 A 或 B。"
        ans = llm(q).strip().upper()
        if ans.startswith("A"):
            return 1
        if ans.startswith("B"):
            return -1
        return 0

    def report(self) -> str:
        return f"ConstitutionalJudge: {len(self.principles)} 条原则，已判定 {self.calls} 次"


# --------------------------------------------------------------------------- #
def rloo_advantages(rewards: torch.Tensor, group: int) -> torch.Tensor:
    """RLOO：leave-one-out baseline——自己的 baseline 排除自己，偏差更小。

    ``rewards``: [G]，``G`` 必须是 ``group`` 的整数倍。
    """
    G = rewards.shape[0]
    if G % group != 0:
        raise ValueError(f"rewards 数量 {G} 必须是 group({group}) 的整数倍")
    r = rewards.view(-1, group)
    total = r.sum(dim=1, keepdim=True)
    others = (total - r) / max(group - 1, 1)
    return (r - others).view(G)


@dataclass
class RewardShaping:
    """奖励塑形：把长度、格式、重复等"软约束"显式加进奖励。"""

    length_penalty: float = 0.0        # 每超出 target_len 一个 token 扣多少
    target_len: int = 128
    format_bonus: float = 0.0
    repeat_penalty: float = 0.0
    clamp_min: float = -1.0
    clamp_max: float = 2.0

    def apply(self, base: float, completion: str,
              format_ok: bool = True) -> float:
        r = base
        n = len(completion)
        if self.length_penalty and n > self.target_len:
            r -= self.length_penalty * (n - self.target_len)
        if self.format_bonus and format_ok:
            r += self.format_bonus
        if self.repeat_penalty:
            r -= self.repeat_penalty * _repeat_ratio(completion)
        return float(max(min(r, self.clamp_max), self.clamp_min))


def _repeat_ratio(text: str, n: int = 8) -> float:
    """8-gram 重复率：最便宜的"复读机"检测器。"""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def shaped_reward(base: float, completion: str,
                  shaping: Optional[RewardShaping] = None,
                  format_ok: bool = True) -> float:
    """便捷函数。"""
    return (shaping or RewardShaping()).apply(base, completion, format_ok)


# --------------------------------------------------------------------------- #
@dataclass
class ThinkingBudget:
    """可控推理预算（Thinking mode / effort 控制）。

    2026 的共识：thinking mode **不是一个新的网络层**，
    它来自后训练、控制 token、chat template 与输出预算。
    所以"控制预算"本质上是**调度与提示**的问题，不是架构问题。
    """

    levels: Dict[str, int] = field(default_factory=lambda: {
        "low": 256, "medium": 1024, "high": 4096,
    })
    default: str = "medium"

    def budget_for(self, level: str) -> int:
        return self.levels.get(level, self.levels[self.default])

    def build_prompt(self, task: str, level: str = "") -> str:
        lv = level or self.default
        return (f"<thinking_budget:{self.budget_for(lv)}>\n"
                f"{task}\n在 {self.budget_for(lv)} 个 token 内完成推理并给出最终答案。")

    def report(self) -> str:
        return f"ThinkingBudget: {self.levels}"
