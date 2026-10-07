"""Agent 的核心数据类型。

设计上向 OpenAI / Anthropic 的消息协议靠拢（role + content + tool_calls），
但额外保留了 **结构化事件流**（``Event``）——这是做可观测性与"可回放调试"的关键：
把一次 Agent 运行完整记录下来，就能离线复现任何一次失败。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = ["Role", "Message", "ToolCall", "ToolResult", "Event", "EventType",
           "LLMResponse", "AgentResult", "new_id"]


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THOUGHT = "thought"
    PLAN = "plan"
    ERROR = "error"
    COMPACTION = "compaction"
    HANDOFF = "handoff"


@dataclass
class Message:
    role: Role
    content: str = ""
    name: Optional[str] = None              # 多智能体：发言者
    tool_calls: List["ToolCall"] = field(default_factory=list)
    tool_call_id: Optional[str] = None      # role=TOOL 时指向哪个调用
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: new_id("call_"))
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    call_id: str = ""
    name: str = ""
    content: str = ""
    is_error: bool = False
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        return Message(role=Role.TOOL, content=self.content,
                       tool_call_id=self.call_id, name=self.name,
                       metadata={"is_error": self.is_error})


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    run_id: str = ""
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "ts": self.ts, "run_id": self.run_id,
                "step": self.step, **self.payload}


@dataclass
class AgentResult:
    answer: str = ""
    success: bool = True
    steps: int = 0
    tool_calls: int = 0
    events: List[Event] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0

    def summary(self) -> str:
        return (f"AgentResult(success={self.success}, steps={self.steps}, "
                f"tool_calls={self.tool_calls}, {self.latency_ms:.0f}ms)")
