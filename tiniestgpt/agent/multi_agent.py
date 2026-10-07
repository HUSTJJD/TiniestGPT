"""多智能体编排：Supervisor / Blackboard / Handoff。

为什么需要多智能体（而不是一个全能 Agent）：
  * **上下文隔离**：每个子 Agent 只看到自己那部分，避免长任务把上下文撑爆；
  * **工具隔离**：给"搜索 Agent"和"代码 Agent"不同的工具集，减少误用；
  * **并行**：互相独立的子任务可以同时跑；
  * **可替换**：子 Agent 可以各自换模型（强模型做规划、小模型做执行）。

三种经典拓扑：
  * **Supervisor（中心化）**：一个调度者分发任务并汇总——最易调试；
  * **Blackboard（共享黑板）**：各 Agent 读写同一块公共状态——适合协作式求解；
  * **Handoff（交接）**：当前 Agent 判断"这不归我管"后把会话整体移交——
    OpenAI Swarm / 客服系统的常见模式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.logging import get_logger
from .observability import Tracer
from .runtime import Agent, AgentConfig
from .tools import ToolRegistry
from .types import AgentResult, EventType, Message, Role, new_id

log = get_logger("tiniestgpt.multi_agent")

__all__ = ["Blackboard", "AgentRole", "Supervisor", "HandoffRouter", "MultiAgentSystem"]


class Blackboard:
    """共享黑板：所有 Agent 都能读写，自带版本日志。"""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.log: List[Dict[str, Any]] = []

    def write(self, key: str, value: Any, author: str = "") -> None:
        self.state[key] = value
        self.log.append({"key": key, "author": author, "ts": time.time(), "value": str(value)[:500]})

    def read(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def snapshot(self) -> str:
        return "\n".join(f"{k}: {str(v)[:300]}" for k, v in self.state.items())


@dataclass
class AgentRole:
    name: str
    description: str = ""
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system_prompt: str = ""
    planner: str = "react"
    max_steps: int = 8

    def build(self, backend, **overrides) -> Agent:
        cfg = AgentConfig(name=self.name, planner=self.planner,
                          system_prompt=self.system_prompt or f"你是 {self.name}：{self.description}",
                          max_steps=self.max_steps, **overrides)
        return Agent(backend=backend, tools=self.tools, cfg=cfg)


class Supervisor:
    """中心化调度：把任务拆给不同角色的 Agent，最后汇总。"""

    def __init__(self, agents: Dict[str, Agent], backend=None, verbose: bool = True) -> None:
        self.agents = agents
        self.backend = backend
        self.verbose = verbose
        self.blackboard = Blackboard()

    def _route(self, task: str) -> List[str]:
        """决定由哪些 Agent 处理（有 LLM 就用 LLM，否则全部参与）。"""
        if self.backend is None or len(self.agents) == 1:
            return list(self.agents)[:1]
        desc = "\n".join(f"- {n}: {getattr(a.cfg, 'name', n)}" for n, a in self.agents.items())
        prompt = (f"任务：{task}\n可用角色：\n{desc}\n"
                  "请只输出需要参与的角色名，用逗号分隔。")
        resp = self.backend.complete([Message(role=Role.USER, content=prompt)],
                                     temperature=0.0, max_tokens=64)
        chosen = [x.strip() for x in resp.content.replace("，", ",").split(",") if x.strip()]
        valid = [c for c in chosen if c in self.agents]
        return valid or list(self.agents)[:1]

    def run(self, task: str, parallel: bool = False) -> AgentResult:
        t0 = time.time()
        names = self._route(task)
        results: Dict[str, AgentResult] = {}
        for n in names:
            if self.verbose:
                log.info("[supervisor] 委派给 %s", n)
            r = self.agents[n].run(task)
            results[n] = r
            self.blackboard.write(f"result:{n}", r.answer, author=n)

        if len(results) == 1:
            only = next(iter(results.values()))
            return AgentResult(answer=only.answer, success=only.success, steps=only.steps,
                               tool_calls=only.tool_calls, latency_ms=(time.time() - t0) * 1000)

        # 多个 Agent 的结果需要"汇总推理"
        if self.backend is None:
            merged = "\n\n".join(f"[{n}] {r.answer}" for n, r in results.items())
            return AgentResult(answer=merged, success=True, latency_ms=(time.time() - t0) * 1000)
        body = "\n\n".join(f"[{n}]: {r.answer}" for n, r in results.items())
        final = self.backend.complete(
            [Message(role=Role.USER, content=f"任务：{task}\n\n各角色结果：\n{body}\n\n请汇总为最终答案：")],
            temperature=0.2, max_tokens=512)
        return AgentResult(answer=final.content.strip(), success=True,
                           steps=sum(r.steps for r in results.values()),
                           tool_calls=sum(r.tool_calls for r in results.values()),
                           latency_ms=(time.time() - t0) * 1000)


class HandoffRouter:
    """交接路由：Agent 输出 ``HANDOFF: <name>`` 时把控制权交出去。"""

    def __init__(self, agents: Dict[str, Agent], default: Optional[str] = None) -> None:
        self.agents = agents
        self.default = default or (next(iter(agents)) if agents else None)

    def run(self, task: str, max_handoffs: int = 4) -> AgentResult:
        current = self.default
        payload = task
        history: List[Message] = [Message(role=Role.USER, content=task)]
        for _ in range(max_handoffs):
            if current is None or current not in self.agents:
                break
            r = self.agents[current].run(payload)
            history.append(Message(role=Role.ASSISTANT, content=r.answer, name=current))
            if "HANDOFF:" in r.answer:
                target = r.answer.split("HANDOFF:")[-1].strip().split()[0].strip(".,;:")
                if target in self.agents:
                    log.info("[handoff] %s -> %s", current, target)
                    payload = f"接手任务：{task}\n上一位的产出：\n{r.answer}"
                    current = target
                    continue
            return r
        return AgentResult(answer="", success=False, error="handoff 链路耗尽")


class MultiAgentSystem:
    """把角色、黑板、调度器组合起来的门面。"""

    def __init__(self, backend=None, verbose: bool = True) -> None:
        self.backend = backend
        self.roles: Dict[str, AgentRole] = {}
        self.agents: Dict[str, Agent] = {}
        self.blackboard = Blackboard()
        self.verbose = verbose

    def register(self, role: AgentRole, backend=None) -> Agent:
        self.roles[role.name] = role
        agent = role.build(backend or self.backend)
        self.agents[role.name] = agent
        return agent

    def supervisor(self) -> Supervisor:
        return Supervisor(self.agents, backend=self.backend, verbose=self.verbose)

    def handoff(self) -> HandoffRouter:
        return HandoffRouter(self.agents)
