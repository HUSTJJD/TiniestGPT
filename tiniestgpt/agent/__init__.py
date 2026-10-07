"""Agentic 框架：工具协议、规划范式、分层记忆、上下文压缩、多智能体、可观测性。

最小可运行示例（不需要任何 LLM，用脚本化后端验证 Agent 逻辑）::

    from tiniestgpt.agent import Agent, AgentConfig, EchoBackend
    from tiniestgpt.agent.builtin_tools import build_default_registry

    backend = EchoBackend(replies=[
        'Thought: 需要计算\\nAction: calculator\\nAction Input: {"expression": "2**10"}',
        "Final Answer: 1024",
    ])
    agent = Agent(backend, build_default_registry())
    print(agent.run("2 的 10 次方是多少？").answer)
"""

from .types import (AgentResult, Event, EventType, LLMResponse, Message, Role, ToolCall,
                    ToolResult)
from .tools import ToolRegistry, ToolSpec, tool, parse_tool_calls
from .memory import MemoryManager, VectorMemory, EpisodicMemory
from .context import ContextManager, ContextConfig
from .planner import ReActPlanner, PlanExecutePlanner, ReflexionPlanner
from .backends import LLMBackend, EchoBackend, LocalEngineBackend, OpenAIBackend
from .runtime import Agent, AgentConfig
from .multi_agent import MultiAgentSystem, Supervisor, Blackboard, HandoffRouter, AgentRole
from .observability import Tracer, Metrics

__all__ = [
    "Agent", "AgentConfig", "AgentResult",
    "Message", "Role", "ToolCall", "ToolResult", "LLMResponse", "Event", "EventType",
    "ToolRegistry", "ToolSpec", "tool", "parse_tool_calls",
    "MemoryManager", "VectorMemory", "EpisodicMemory",
    "ContextManager", "ContextConfig",
    "ReActPlanner", "PlanExecutePlanner", "ReflexionPlanner",
    "LLMBackend", "EchoBackend", "LocalEngineBackend", "OpenAIBackend",
    "MultiAgentSystem", "Supervisor", "Blackboard", "HandoffRouter", "AgentRole",
    "Tracer", "Metrics",
]
