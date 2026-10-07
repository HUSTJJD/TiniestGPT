"""LLM 后端：把"谁来当大脑"这件事抽象掉。

这样同一个 Agent 可以：
  * 用本地 TiniestGPT 引擎（离线、可复现）；
  * 用任何 OpenAI 兼容服务（含本项目自己的 ``serve`` 端点）；
  * 用脚本化的 Echo 后端做**单元测试**（不依赖模型也能验证 Agent 逻辑正确性）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from ..common.registry import BACKEND
from .types import LLMResponse, Message, Role

__all__ = ["LLMBackend", "EchoBackend", "LocalEngineBackend", "OpenAIBackend",
           "format_messages"]


def format_messages(messages: List[Message]) -> str:
    """把消息列表渲染成纯文本提示（小模型没有 chat template 时的通用做法）。"""
    out = []
    for m in messages:
        role = m.role.value.upper()
        if m.role == Role.TOOL:
            out.append(f"OBSERVATION[{m.name or 'tool'}]: {m.content}")
        else:
            out.append(f"{role}: {m.content}")
    return "\n".join(out) + "\nASSISTANT:"


class LLMBackend:
    name = "base"

    def complete(self, messages: List[Message], tools: Optional[List[Dict]] = None,
                 temperature: float = 0.7, max_tokens: int = 256,
                 stop: Optional[List[str]] = None) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class EchoBackend(LLMBackend):
    """脚本化后端：按预设的回复序列依次返回（用于测试与演示）。"""

    name = "echo"

    def __init__(self, replies: Optional[List[str]] = None, final: str = "done") -> None:
        self.replies = list(replies or [])
        self.final = final
        self.i = 0

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=256, stop=None) -> LLMResponse:
        from .tools import parse_tool_calls

        text = self.replies[self.i] if self.i < len(self.replies) else f"Final Answer: {self.final}"
        self.i += 1
        return LLMResponse(content=text, tool_calls=parse_tool_calls(text))


# --------------------------------------------------------------------------- #
class LocalEngineBackend(LLMBackend):
    """用本项目推理引擎做大脑。"""

    name = "local"

    def __init__(self, engine, tokenizer=None, use_tool_protocol: bool = True) -> None:
        self.engine = engine
        self.tokenizer = tokenizer or getattr(engine, "tokenizer", None)
        self.use_tool_protocol = use_tool_protocol

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=256,
                 stop=None) -> LLMResponse:
        from ..inference.sampler import SamplingParams
        from .tools import parse_tool_calls

        prompt = format_messages(messages)
        params = SamplingParams(temperature=max(temperature, 1e-3), max_tokens=max_tokens,
                                top_p=0.95)
        outs = self.engine.generate(prompt, params)
        text = outs[0].text if outs else ""
        calls = parse_tool_calls(text) if self.use_tool_protocol else []
        return LLMResponse(content=text, tool_calls=calls,
                           prompt_tokens=outs[0].prompt_tokens if outs else 0,
                           completion_tokens=outs[0].completion_tokens if outs else 0)


# --------------------------------------------------------------------------- #
class OpenAIBackend(LLMBackend):
    """任意 OpenAI 兼容端点（可指向 ``python -m tiniestgpt.cli serve``）。"""

    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "tiniestgpt", api_key: Optional[str] = None,
                 timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=256,
                 stop=None) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = stop
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        choice = obj["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        calls = []
        from .types import ToolCall

        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
        usage = obj.get("usage", {})
        return LLMResponse(content=content, tool_calls=calls,
                           finish_reason=choice.get("finish_reason", "stop"),
                           prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0), raw=obj)


BACKEND._items.setdefault("echo", EchoBackend)
BACKEND._items.setdefault("local", LocalEngineBackend)
BACKEND._items.setdefault("openai", OpenAIBackend)
