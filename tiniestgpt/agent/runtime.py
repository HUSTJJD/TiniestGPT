"""Agent 运行时：把 模型 / 工具 / 记忆 / 上下文 / 规划 / 可观测 串成一个闭环。

一次 ``run()`` 的骨架::

    for step in range(max_steps):
        messages = context.build(system, memory, history, retrieved)   # 上下文管理
        resp     = backend.complete(messages, tools)                   # 模型推理
        step_out = planner.parse(resp.content)                         # 规划解析
        if step_out.tool_calls:
            执行工具 → 结果写回 history + memory                        # 行动与观察
        else:
            return final answer                                        # 终止

鲁棒性设计：
  * 工具失败**不中断**，把错误文本回灌给模型让它自我修正；
  * 连续失败会触发 Reflexion 反思并重试；
  * 每一步都写 Trace，运行结束写情景记忆（供下次复用）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.logging import get_logger
from .backends import EchoBackend, LLMBackend
from .context import ContextConfig, ContextManager
from .memory import MemoryManager
from .observability import Metrics, Tracer
from .planner import AgentStep, PlanExecutePlanner, Planner, ReflexionPlanner, ReActPlanner
from .tools import ToolRegistry
from .types import (AgentResult, EventType, LLMResponse, Message, Role, ToolCall, ToolResult,
                    new_id)

log = get_logger("tiniestgpt.agent")

__all__ = ["AgentConfig", "Agent"]


@dataclass
class AgentConfig:
    max_steps: int = 12
    planner: str = "react"               # react | plan_execute | reflexion
    temperature: float = 0.3
    max_tokens: int = 256
    max_retries: int = 2                 # Reflexion 重试次数
    parallel_tools: bool = True          # 一步内多个工具是否并发执行
    enable_memory: bool = True
    retrieve_k: int = 3
    verbose: bool = True
    trace_path: Optional[str] = None
    name: str = "agent"
    system_prompt: str = "你是一个有用、精确、会主动使用工具的 AI 助手。"


class Agent:
    def __init__(self, backend: LLMBackend, tools: Optional[ToolRegistry] = None,
                 cfg: Optional[AgentConfig] = None, memory: Optional[MemoryManager] = None,
                 context: Optional[ContextManager] = None, planner: Optional[Planner] = None,
                 tracer: Optional[Tracer] = None, metrics: Optional[Metrics] = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.backend = backend
        self.tools = tools or ToolRegistry()
        self.memory = memory or MemoryManager()
        self.context = context or ContextManager()
        self.planner = planner or self._make_planner(self.cfg.planner)
        self.metrics = metrics or Metrics()
        self.tracer = tracer
        self.run_id = new_id("run_")

    def _make_planner(self, name: str) -> Planner:
        return {"react": ReActPlanner, "plan_execute": PlanExecutePlanner,
                "reflexion": ReflexionPlanner}.get(name, ReActPlanner)()

    # ------------------------------------------------------------------ #
    @property
    def system_prompt(self) -> str:
        return self.cfg.system_prompt + "\n\n" + self.planner.system_prompt(self.tools.describe())

    # ------------------------------------------------------------------ #
    def run(self, task: str) -> AgentResult:
        t0 = time.time()
        tracer = self.tracer or Tracer(run_id=self.run_id, path=self.cfg.trace_path)
        tracer.emit(EventType.RUN_START, task=task, agent=self.cfg.name)

        history: List[Message] = [Message(role=Role.USER, content=task)]
        result = AgentResult()
        retrieved: List[str] = []
        if self.cfg.enable_memory:
            hits = self.memory.recall(task, k=self.cfg.retrieve_k)
            retrieved = [h.text for h in hits]

        for attempt in range(self.cfg.max_retries + 1):
            outcome = self._inner_loop(task, history, retrieved, tracer, attempt)
            result = outcome
            if outcome.success:
                break
            if attempt < self.cfg.max_retries and isinstance(self.planner, ReflexionPlanner):
                self._reflect(task, history, outcome.error or "未完成", tracer)
                history = [Message(role=Role.USER, content=task)]   # 重新开头，但记忆里已有反思
            else:
                break

        result.latency_ms = (time.time() - t0) * 1000
        result.events = tracer.events
        if self.cfg.enable_memory:
            self.memory.episodic.record(
                task, trajectory="\n".join(m.content[:200] for m in history[-8:]),
                success=result.success, score=1.0 if result.success else 0.0)
            self.memory.observe(f"任务: {task}\n结果: {result.answer[:400]}")
        tracer.emit(EventType.RUN_END, success=result.success,
                    steps=result.steps, latency_ms=result.latency_ms)
        if self.cfg.verbose:
            log.info("%s", result.summary())
        return result

    # ------------------------------------------------------------------ #
    def _inner_loop(self, task: str, history: List[Message], retrieved: List[str],
                    tracer: Tracer, attempt: int) -> AgentResult:
        result = AgentResult(success=False, error="达到最大步数仍未完成")
        for step in range(self.cfg.max_steps):
            result.steps += 1
            messages = self.context.build(self.system_prompt, self.memory.context_text(),
                                          history, retrieved)
            tracer.emit(EventType.LLM_CALL, step=step, n_messages=len(messages))
            t0 = time.time()
            try:
                resp: LLMResponse = self.backend.complete(
                    messages, tools=self.tools.schemas(),
                    temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
            except Exception as exc:
                tracer.emit(EventType.ERROR, step=step, error=str(exc))
                result.error = f"LLM 调用失败: {exc}"
                return result
            self.metrics.record_time("llm_latency", (time.time() - t0) * 1000)
            self.metrics.incr("llm_calls")
            tracer.emit(EventType.LLM_RESPONSE, step=step, content=resp.content[:500])

            parsed: AgentStep = self.planner.parse(resp.content)
            if parsed.thought and self.cfg.verbose:
                log.info("  [thought] %s", parsed.thought[:200])
                tracer.emit(EventType.THOUGHT, step=step, thought=parsed.thought[:500])

            if parsed.is_final:
                result.answer = parsed.final_answer or ""
                result.success = True
                result.error = None
                history.append(Message(role=Role.ASSISTANT, content=resp.content))
                return result

            if not parsed.tool_calls:
                # 模型没给工具调用也没给最终答案 → 把原文当作答案，避免死循环
                result.answer = resp.content.strip()
                result.success = True
                result.error = None
                return result

            history.append(Message(role=Role.ASSISTANT, content=resp.content,
                                   tool_calls=parsed.tool_calls))
            for tc in parsed.tool_calls:
                res = self._execute_tool(tc, tracer, step)
                result.tool_calls += 1
                content = self.context.truncate_tool_result(res.content)
                history.append(Message(role=Role.TOOL, content=content,
                                       tool_call_id=tc.id, name=tc.name,
                                       metadata={"is_error": res.is_error}))
                if self.cfg.enable_memory:
                    self.memory.observe(f"工具 {tc.name} 返回: {content[:300]}", tool=tc.name)
                if self.cfg.verbose:
                    log.info("  [tool] %s -> %s", tc.name, content[:160].replace("\n", " "))
        return result

    # ------------------------------------------------------------------ #
    def _execute_tool(self, tc: ToolCall, tracer: Tracer, step: int) -> ToolResult:
        tracer.emit(EventType.TOOL_CALL, step=step, name=tc.name, arguments=tc.arguments)
        t0 = time.time()
        try:
            res = self.tools.invoke(tc.name, tc.arguments, call_id=tc.id)
        except KeyError as exc:
            res = ToolResult(call_id=tc.id, name=tc.name, content=str(exc), is_error=True)
        res.latency_ms = (time.time() - t0) * 1000
        self.metrics.incr("tool_calls")
        self.metrics.incr(f"tool.{tc.name}")
        self.metrics.record_time("tool_latency", res.latency_ms)
        tracer.emit(EventType.TOOL_RESULT, step=step, name=tc.name,
                    content=res.content[:500], is_error=res.is_error)
        return res

    # ------------------------------------------------------------------ #
    def _reflect(self, task: str, history: List[Message], error: str, tracer: Tracer) -> None:
        assert isinstance(self.planner, ReflexionPlanner)
        prompt = self.planner.reflect_prompt(
            task, "\n".join(m.content[:300] for m in history), error)
        resp = self.backend.complete([Message(role=Role.USER, content=prompt)],
                                     temperature=0.2, max_tokens=200)
        self.memory.remember(f"[反思] {resp.content}", kind="reflection")
        tracer.emit(EventType.THOUGHT, payload={"reflection": resp.content[:500]})
        if self.cfg.verbose:
            log.info("  [reflection] %s", resp.content[:200])

    # ------------------------------------------------------------------ #
    def run_plan_execute(self, task: str) -> AgentResult:
        """Plan-and-Execute 范式：先规划，再逐步执行。"""
        if not isinstance(self.planner, PlanExecutePlanner):
            self.planner = PlanExecutePlanner()
        tracer = self.tracer or Tracer(run_id=self.run_id, path=self.cfg.trace_path)
        plan_prompt = self.planner.plan_prompt(task, self.tools.describe())
        resp = self.backend.complete([Message(role=Role.USER, content=plan_prompt)],
                                     temperature=0.2, max_tokens=256)
        plan = self.planner.parse_plan(resp.content)
        tracer.emit(EventType.PLAN, plan=plan)
        if not plan:
            return self.run(task)          # 规划失败则退回 ReAct

        observations: List[str] = []
        history: List[Message] = [Message(role=Role.USER, content=task)]
        for idx, step_text in enumerate(plan):
            history.append(Message(role=Role.USER,
                                   content=self.planner.step_prompt(plan, idx, "\n".join(observations))))
            sub = self._inner_loop(step_text, history, [], tracer, idx)
            observations.append(f"步骤{idx+1} ({step_text}): {sub.answer[:300]}")
            if not sub.success:
                observations.append(f"[警告] 步骤{idx+1} 未完成: {sub.error}")
        final = self.backend.complete(
            [Message(role=Role.USER, content=f"任务：{task}\n执行记录：\n" + "\n".join(observations)
                     + "\n\n请给出最终答案：")],
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        return AgentResult(answer=final.content.strip(), success=True, steps=len(plan),
                           events=tracer.events)
