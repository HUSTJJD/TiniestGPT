"""成本护栏 + HITL 人工审批 + 权限分级。

长程 Agent 最容易出的三种事故，全都是"跑起来就停不下来"：

1. **循环/挂起**：Agent 在两个动作之间反复横跳，token 无限烧；
2. **预算穿透**：没有 token/金额上限，一次任务烧掉几千块；
3. **越权操作**：删除文件、发邮件、改数据库——没有权限分级就敢做。

护栏的设计原则：**可解释、可覆盖、默认保守**。

* :class:`CostGuard` —— token/步数/时间三重预算 + 循环检测；
* :class:`Permission` / :class:`HITL` —— 危险动作需要人工批准。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

__all__ = ["Budget", "CostGuard", "LoopDetector", "Permission", "HITL",
           "GuardDecision"]


@dataclass
class Budget:
    max_tokens: int = 100_000
    max_steps: int = 30
    max_seconds: float = 300.0
    max_cost: float = 5.0            # 金额上限（按 price 换算）


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    used: Dict[str, float] = field(default_factory=dict)


class LoopDetector:
    """循环检测：同一"动作指纹"重复出现就报警。

    指纹 = 工具名 + 参数哈希。只看输出文本会被"每次措辞略不同"骗过去。
    """

    def __init__(self, window: int = 8, repeat_threshold: int = 3) -> None:
        self.window = window
        self.threshold = repeat_threshold
        self.recent: List[str] = []
        self.loops = 0

    @staticmethod
    def fingerprint(tool: str, args: str) -> str:
        return tool + ":" + hashlib.md5(args.encode("utf-8")).hexdigest()[:8]

    def observe(self, tool: str, args: str) -> bool:
        fp = self.fingerprint(tool, args)
        self.recent.append(fp)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        if self.recent.count(fp) >= self.threshold:
            self.loops += 1
            return True
        return False

    def report(self) -> str:
        return f"LoopDetector: 检测到 {self.loops} 次循环（阈值 {self.threshold}/{self.window}）"


class CostGuard:
    """三重预算守卫 + 循环检测。"""

    def __init__(self, budget: Optional[Budget] = None,
                 price_per_1k: float = 0.002,
                 loop: Optional[LoopDetector] = None) -> None:
        self.budget = budget or Budget()
        self.price_per_1k = price_per_1k
        self.loop = loop or LoopDetector()
        self.tokens = 0
        self.steps = 0
        self.t0 = time.time()
        self.stopped_reason = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def cost(self) -> float:
        return self.tokens / 1000.0 * self.price_per_1k

    def usage(self) -> Dict[str, float]:
        return {"tokens": float(self.tokens), "steps": float(self.steps),
                "seconds": self.elapsed, "cost": self.cost}

    def check(self, tool: str, args: str = "") -> GuardDecision:
        b = self.budget
        if self.loop.observe(tool, args):
            self.stopped_reason = "检测到动作循环"
            return GuardDecision(False, self.stopped_reason, self.usage())
        if self.tokens >= b.max_tokens:
            self.stopped_reason = f"token 预算耗尽 ({self.tokens}/{b.max_tokens})"
        elif self.steps >= b.max_steps:
            self.stopped_reason = f"步数预算耗尽 ({self.steps}/{b.max_steps})"
        elif self.elapsed >= b.max_seconds:
            self.stopped_reason = f"时间预算耗尽 ({self.elapsed:.0f}s/{b.max_seconds:.0f}s)"
        elif self.cost >= b.max_cost:
            self.stopped_reason = f"金额预算耗尽 ({self.cost:.2f}/{b.max_cost})"
        if self.stopped_reason:
            return GuardDecision(False, self.stopped_reason, self.usage())
        return GuardDecision(True, "", self.usage())

    def consume(self, tokens: int = 0, steps: int = 1) -> None:
        self.tokens += int(tokens)
        self.steps += int(steps)

    def report(self) -> str:
        u = self.usage()
        return (f"CostGuard: {u['tokens']:.0f} tokens / {u['steps']:.0f} 步 / "
                f"{u['seconds']:.1f}s / ${u['cost']:.4f}；{self.loop.report()}"
                + (f"｜停止原因：{self.stopped_reason}" if self.stopped_reason else ""))


# --------------------------------------------------------------------------- #
@dataclass
class Permission:
    """权限分级：把工具分成"可直接做 / 需批准 / 禁止"三档。"""

    allow: Set[str] = field(default_factory=lambda: {"think", "now", "calculator",
                                                     "read_file", "list_dir", "rag_search"})
    require_approval: Set[str] = field(default_factory=lambda: {"write_file", "python_repl",
                                                                "web_search"})
    deny: Set[str] = field(default_factory=lambda: {"python_sandbox"})

    def level_of(self, tool: str) -> str:
        if tool in self.deny:
            return "deny"
        if tool in self.require_approval:
            return "approval"
        if tool in self.allow:
            return "allow"
        return "approval"          # 未知工具默认要批准（保守默认）

    def report(self) -> str:
        return (f"Permission: allow={len(self.allow)}, "
                f"approval={len(self.require_approval)}, deny={len(self.deny)}")


class HITL:
    """Human-in-the-Loop：危险动作停下来问人。

    :param approver: ``(tool, args) -> bool``；None 时用交互式 stdin
    """

    def __init__(self, permission: Optional[Permission] = None,
                 approver: Optional[Callable[[str, str], bool]] = None,
                 auto_approve: bool = False) -> None:
        self.permission = permission or Permission()
        self.approver = approver
        self.auto_approve = auto_approve
        self.pending: List[Dict[str, str]] = []
        self.approved = 0
        self.rejected = 0

    def gate(self, tool: str, args: str = "") -> GuardDecision:
        lvl = self.permission.level_of(tool)
        if lvl == "deny":
            return GuardDecision(False, f"工具 {tool} 被明确禁止")
        if lvl == "allow":
            return GuardDecision(True, "")
        if self.auto_approve:
            self.approved += 1
            return GuardDecision(True, "auto-approved")
        ok = self.approver(tool, args) if self.approver else self._ask(tool, args)
        if ok:
            self.approved += 1
            return GuardDecision(True, "human approved")
        self.rejected += 1
        self.pending.append({"tool": tool, "args": args})
        return GuardDecision(False, "人工拒绝")

    @staticmethod
    def _ask(tool: str, args: str) -> bool:
        try:
            ans = input(f"[HITL] 是否允许调用 {tool}({args[:60]})? [y/N] ").strip().lower()
        except Exception:
            return False          # 非交互环境一律拒绝
        return ans.startswith("y")

    def report(self) -> str:
        return (f"HITL: 批准 {self.approved}，拒绝 {self.rejected}，"
                f"待处理 {len(self.pending)}；{self.permission.report()}")
