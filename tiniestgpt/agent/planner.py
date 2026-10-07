"""Planner：决定"模型下一步该怎么思考"。

三种主流范式，各自的适用场景：

* **ReAct**（Thought → Action → Observation 循环）
  最通用、最省 token。适合"边查边想"的工具型任务。
* **Plan-and-Execute**（先规划全局步骤，再逐步执行）
  适合**多步依赖**的长任务；规划与执行分离后可以各用不同的模型
  （规划用强模型，执行用快模型），这是降本的常用手段。
* **Reflexion**（失败后自我反思，把反思写进记忆再重试）
  在"容易反复犯同样错误"的场景（代码、数学）收益最大。

实现上，Planner 只负责两件事：**生成提示词** 与 **解析模型输出**。
执行与记忆由 runtime 负责——关注点分离让三者可以自由组合。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..common.registry import PLANNER
from .types import ToolCall

__all__ = ["AgentStep", "Planner", "ReActPlanner", "PlanExecutePlanner", "ReflexionPlanner"]


@dataclass
class AgentStep:
    thought: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None and not self.tool_calls


class Planner:
    name = "base"

    # ------------------------------------------------------------------ #
    def system_prompt(self, tool_descriptions: str) -> str:
        raise NotImplementedError

    def parse(self, text: str) -> AgentStep:
        raise NotImplementedError

    def observation_prompt(self, name: str, content: str) -> str:
        return f"Observation ({name}): {content}"


# --------------------------------------------------------------------------- #
class ReActPlanner(Planner):
    """Thought / Action / Observation 循环。"""

    name = "react"

    def system_prompt(self, tool_descriptions: str) -> str:
        return (
            "你是一个会使用工具的 AI 助手。按如下格式严格输出：\n"
            "Thought: <你对当前情况的简短分析>\n"
            "Action: <工具名>\n"
            "Action Input: <JSON 参数对象>\n"
            "或者当你已经有答案时：\n"
            "Final Answer: <最终答案>\n\n"
            "可用工具：\n" + tool_descriptions + "\n"
            "注意：Action 必须是上面列出的工具之一；Action Input 必须是合法 JSON。"
        )

    def parse(self, text: str) -> AgentStep:
        step = AgentStep()
        m = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|$)", text, re.S)
        if m:
            step.thought = m.group(1).strip()

        fm = re.search(r"Final Answer:\s*(.*)$", text, re.S)
        if fm:
            step.final_answer = fm.group(1).strip()
            return step

        am = re.search(r"Action:\s*([A-Za-z_][\w]*)\s*\n\s*Action Input:\s*(.*?)(?=\n(?:Action|Final Answer|Observation):|$)",
                       text, re.S)
        if am:
            name = am.group(1).strip()
            raw = am.group(2).strip().strip("`")
            args = self._parse_args(raw)
            step.tool_calls.append(ToolCall(name=name, arguments=args))
            return step

        # 兼容 <tool_call> 协议
        from .tools import parse_tool_calls

        calls = parse_tool_calls(text)
        if calls:
            step.tool_calls = calls
        else:
            step.final_answer = text.strip()
        return step

    @staticmethod
    def _parse_args(raw: str) -> Dict:
        import json

        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"input": v}
        except Exception:
            return {"input": raw}


# --------------------------------------------------------------------------- #
class PlanExecutePlanner(ReActPlanner):
    """先规划再执行：把任务拆成可执行的有序步骤。"""

    name = "plan_execute"

    def plan_prompt(self, task: str, tool_descriptions: str) -> str:
        return (
            "请把下面的任务拆成 3~6 个有序步骤，输出严格 JSON 数组，不要有多余文字：\n"
            '示例: ["步骤一", "步骤二"]\n\n'
            f"可用工具：\n{tool_descriptions}\n\n任务：{task}"
        )

    def parse_plan(self, text: str) -> List[str]:
        import json

        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            v = json.loads(m.group(0))
        except Exception:
            return []
        return [str(x) for x in v if isinstance(x, (str, int, float))]

    def step_prompt(self, plan: List[str], idx: int, observations: str) -> str:
        return (
            f"整体计划：\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)) + "\n\n"
            f"现在执行第 {idx+1} 步：{plan[idx]}\n"
            f"已有的观察：\n{observations}\n"
            "请输出 Thought / Action / Action Input，或直接 Final Answer。"
        )


# --------------------------------------------------------------------------- #
class ReflexionPlanner(ReActPlanner):
    """在 ReAct 之上加一层"失败反思"。"""

    name = "reflexion"

    def reflect_prompt(self, task: str, trajectory: str, error: str) -> str:
        return (
            "下面是一次失败的任务执行记录。请用 3~5 句话回答：\n"
            "1) 失败的直接原因是什么？\n"
            "2) 下次应该采取什么不同的策略？\n"
            "只输出反思内容，不要输出 JSON。\n\n"
            f"任务：{task}\n\n轨迹：\n{trajectory[:2000]}\n\n错误：{error}"
        )


PLANNER._items.setdefault("react", ReActPlanner)
PLANNER._items.setdefault("plan_execute", PlanExecutePlanner)
PLANNER._items.setdefault("reflexion", ReflexionPlanner)
