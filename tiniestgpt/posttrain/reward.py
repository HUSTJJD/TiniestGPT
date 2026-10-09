"""奖励：RL 闭环里最容易"想当然"的一环。

2026 年的共识（DeepSeek-R1 / Qwen3 / Kimi K2 一致）：

1. **可验证奖励（RLVR）优先**。数学/代码/工具调用这类任务的对错
   由规则或执行结果说了算，**不需要再训一个裁判模型**，
   既省显存又天然免疫 reward hacking;
2. **奖励模型（RM）只用来表达主观偏好**（helpful / harmless / 风格），
   用 Bradley-Terry 成对损失训练;
3. **混合奖励是最佳实践**：RM（约 70%）+ 规则（约 30%），
   兼得主观偏好与客观正确。

本模块同时提供三者的最小可用实现。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RuleReward", "RewardModel", "HybridReward", "bradley_terry_loss",
           "reward_from_rules"]


# --------------------------------------------------------------------------- #
#  可验证奖励（RLVR）
# --------------------------------------------------------------------------- #
def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


class RuleReward:
    """规则奖励：**可验证**，因此不会有"裁判模型被骗"的问题。

    :param kind:
      * ``exact_int``   —— 抽取第一个整数与参考答案比对（算术）
      * ``exact_match`` —— 归一化字符串相等
      * ``contains``    —— 参考答案是否出现在输出里
      * ``json_valid``  —— 能否解析出 JSON 且必填字段齐全
      * ``unit_test``   —— 用 ``exec`` 跑断言（代码任务；会走 :class:`沙箱` 的超时保护）
    :param format_bonus: 输出满足"先给结论"等格式要求时的小额加成（防长度爆炸）
    """

    KINDS = ("exact_int", "exact_match", "contains", "json_valid", "unit_test")

    def __init__(self, kind: str = "exact_int", format_bonus: float = 0.0,
                 max_length_penalty: float = 0.0, max_len: int = 128) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"未知规则奖励: {kind}（可选 {self.KINDS}）")
        self.kind = kind
        self.format_bonus = format_bonus
        self.max_length_penalty = max_length_penalty
        self.max_len = max_len

    # ------------------------------------------------------------------ #
    def __call__(self, prompt: str, completion: str, answer: Any = None) -> float:
        r = 0.0
        if self.kind == "exact_int":
            got = _extract_int(completion)
            r = 1.0 if (got is not None and answer is not None and got == int(answer)) else 0.0
        elif self.kind == "exact_match":
            r = 1.0 if completion.strip().lower().startswith(str(answer).strip().lower()) else 0.0
        elif self.kind == "contains":
            r = 1.0 if str(answer) in completion else 0.0
        elif self.kind == "json_valid":
            r = self._json_score(completion, answer)
        elif self.kind == "unit_test":
            r = self._unit_test(completion, answer)
        if self.format_bonus and re.match(r"^\s*(Answer|答案)\s*[:：]", completion):
            r += self.format_bonus
        if self.max_length_penalty and len(completion.split()) > self.max_len:
            r -= self.max_length_penalty
        return float(r)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _json_score(completion: str, schema: Any) -> float:
        s, e = completion.find("{"), completion.rfind("}")
        if s < 0 or e <= s:
            return 0.0
        try:
            obj = json.loads(completion[s:e + 1])
        except Exception:
            return 0.0
        if not isinstance(obj, dict) or not isinstance(schema, dict):
            return 1.0 if isinstance(obj, dict) else 0.0
        ok = sum(1 for k, t in schema.items() if k in obj)
        return ok / max(len(schema), 1)

    @staticmethod
    def _unit_test(completion: str, tests: Any) -> float:
        """执行断言型单测。**仅限受控环境**——真实系统必须放进沙箱。"""
        if not tests:
            return 0.0
        code = completion
        if "```" in code:
            blocks = re.findall(r"```(?:python)?\s*(.*?)```", code, re.S)
            if blocks:
                code = blocks[0]
        ns: Dict[str, Any] = {}
        try:
            exec(compile(code, "<rlvr>", "exec"), ns)      # noqa: S102 - 教学用途
        except Exception:
            return 0.0
        passed = 0
        for t in (tests if isinstance(tests, list) else [tests]):
            try:
                exec(compile(str(t), "<test>", "exec"), ns)  # noqa: S102
                passed += 1
            except Exception:
                pass
        return passed / len(tests if isinstance(tests, list) else [tests])


def reward_from_rules(rule: RuleReward, completions: Sequence[str], answer: Any,
                      prompt: str = "") -> torch.Tensor:
    """把规则奖励应用到一组采样上 → ``[G]`` 张量。"""
    return torch.tensor([rule(prompt, c, answer) for c in completions], dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  奖励模型（主观偏好）
# --------------------------------------------------------------------------- #
class RewardModel(nn.Module):
    """在策略模型的隐藏状态上接一个标量打分头。

    刻意做得极小：一个线性头 + 最后一个非 padding token 的池化。
    工业界的 RM 与策略同规模（7B 配 7B），这里是为了让"RM 到底是什么"
    保持可见——它就是一个把序列映射成标量的函数。
    """

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1, bias=False))

    def forward(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``hidden: [N,T,D]`` → ``[N]``"""
        s = self.score(hidden).squeeze(-1)                       # [N,T]
        if mask is not None:
            m = mask.float()
            s = (s * m).sum(-1) / m.sum(-1).clamp(min=1)
        else:
            s = s[:, -1]
        return s


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """成对偏好损失（Bradley-Terry）：``-log σ(r_w - r_l)``。

    它隐含的假设是"人类偏好服从 Boltzmann 分布"——这个假设并不总成立，
    但它在工程上极其稳定，是 RLHF / RLAIF 的通用底座。
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# --------------------------------------------------------------------------- #
@dataclass
class HybridReward:
    """混合奖励：``w_rm · RM + w_rule · 规则``（2026 最佳实践约 7:3）。"""

    rule: RuleReward = field(default_factory=RuleReward)
    rm: Optional[RewardModel] = None
    w_rm: float = 0.0
    w_rule: float = 1.0

    def __call__(self, prompt: str, completion: str, answer: Any = None,
                 hidden: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None) -> float:
        r = self.w_rule * self.rule(prompt, completion, answer)
        if self.rm is not None and self.w_rm > 0 and hidden is not None:
            with torch.no_grad():
                r = r + self.w_rm * float(self.rm(hidden, mask).mean().item())
        return float(r)
