"""Agent 任务评测：从"能不能用"到"靠不靠谱"。

2026 年 Agent 评测的范式转移，核心是三件事：

1. **可靠性 > 峰值表现**
   跑通 1 次不算数。要用 **Pass^k**：同一任务独立跑 k 次，
   **k 次全通过**才算通过。这会把"偶尔蒙对"的模型直接筛掉。
2. **成本是硬指标**
   两个 Agent 都完成了任务，但一个用了 3 步、另一个用了 30 步——
   后者在生产上就是不可用的。必须追踪 token 与步数。
3. **轨迹也要看**
   只看最终答案会漏掉"过程完全错误但答案蒙对"的情况，
   所以要检查轨迹（是否真的用了工具、是否忽略了工具返回）。

本模块提供 :class:`AgentEvaluator`，支持 Pass^k、成本统计与轨迹检查。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["AgentTask", "AgentRun", "AgentEvaluator", "pass_at_k", "pass_k_estimator"]


@dataclass
class AgentTask:
    name: str
    prompt: str
    check: Callable[[str], bool]          # 最终答案是否通过
    expect_tool: Optional[str] = None     # 期望用到的工具（None = 不限）
    max_steps: int = 20


@dataclass
class AgentRun:
    task: str
    success: bool
    steps: int = 0
    tokens: int = 0
    seconds: float = 0.0
    used_tool: bool = False
    ignored_tool_output: bool = False
    answer: str = ""


def pass_at_k(n: int, c: int, k: int) -> float:
    """经典的 pass@k 无偏估计：``1 - C(n-c, k) / C(n, k)``。"""
    if n < k or c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_k_estimator(runs: Sequence[bool], k: int) -> float:
    """Pass^k：**k 次独立运行全部成功**的比例。

    与 pass@k 的区别：pass@k 是"k 次里至少一次成功"（宽松），
    pass^k 是"k 次里每次都成功"（严格）——后者才对应生产可靠性。
    """
    if len(runs) < k or k <= 0:
        return 0.0
    groups = [runs[i:i + k] for i in range(0, len(runs) - k + 1, k)]
    if not groups:
        return 0.0
    return sum(1.0 for g in groups if all(g)) / len(groups)


class AgentEvaluator:
    """跑一组任务、重复 k 次，给出成功率 / Pass^k / 成本三张表。"""

    def __init__(self, tasks: Sequence[AgentTask], repeats: int = 3) -> None:
        self.tasks = list(tasks)
        self.repeats = repeats
        self.runs: List[AgentRun] = []

    def run(self, agent_fn: Callable[[str], Dict]) -> "AgentEvaluator":
        """``agent_fn(prompt) -> {"answer": str, "steps": int, "tokens": int,
        "seconds": float, "used_tool": bool, "ignored_tool_output": bool}``"""
        for t in self.tasks:
            for _ in range(self.repeats):
                out = agent_fn(t.prompt) or {}
                ok = bool(out.get("answer")) and t.check(out.get("answer", ""))
                self.runs.append(AgentRun(
                    task=t.name, success=ok,
                    steps=int(out.get("steps", 0)), tokens=int(out.get("tokens", 0)),
                    seconds=float(out.get("seconds", 0.0)),
                    used_tool=bool(out.get("used_tool", False)),
                    ignored_tool_output=bool(out.get("ignored_tool_output", False)),
                    answer=str(out.get("answer", "")),
                ))
        return self

    # ------------------------------------------------------------------ #
    def by_task(self) -> Dict[str, List[bool]]:
        d: Dict[str, List[bool]] = {}
        for r in self.runs:
            d.setdefault(r.task, []).append(r.success)
        return d

    def summary(self) -> Dict[str, float]:
        if not self.runs:
            return {}
        n = len(self.runs)
        succ = [r.success for r in self.runs]
        bt = self.by_task()
        pass_k = {k: pass_k_estimator(v, min(self.repeats, len(v)))
                  for k, v in bt.items()}
        return {
            "runs": float(n),
            "success_rate": sum(succ) / n,
            "pass^k_all": sum(1.0 for v in bt.values() if all(v)) / max(len(bt), 1),
            "avg_steps": sum(r.steps for r in self.runs) / n,
            "avg_tokens": sum(r.tokens for r in self.runs) / n,
            "cost_per_success": sum(r.tokens for r in self.runs) / max(sum(succ), 1),
            "tool_ignored_rate": sum(r.ignored_tool_output for r in self.runs) / n,
            "tasks": float(len(bt)),
        }

    def table(self) -> str:
        d = self.by_task()
        lines = ["%-16s %8s %8s" % ("task", "success", "pass^k")]
        for name, res in d.items():
            lines.append("%-16s %8.2f %8.2f" % (
                name, sum(res) / len(res), pass_k_estimator(res, len(res))))
        s = self.summary()
        if s:
            lines.append("-" * 36)
            lines.append(f"平均成功率 {s['success_rate']:.2%} | "
                         f"平均步数 {s['avg_steps']:.1f} | "
                         f"每成功一次 {s['cost_per_success']:.0f} tokens")
        return "\n".join(lines)
