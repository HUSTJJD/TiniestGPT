"""工具协议：把 Python 函数变成 LLM 可调用的结构化工具。

关键设计：
  * **单一事实来源**：一份 Python 类型标注 + docstring 同时生成
    JSON Schema（给模型看）与运行时校验（给人看），避免"文档与实现不一致"；
  * **函数是纯函数优先**：工具应该是可重放、可缓存、无隐藏状态的，
    这样失败重试与离线回放才是安全的；
  * **统一的错误返回**：工具抛异常不应该让整个 Agent 崩，
    而是把错误文本回灌给模型——**让模型自己修** 是 Agent 鲁棒性的核心。
"""

from __future__ import annotations

import asyncio
import inspect
import time
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin, get_type_hints

from ..common.registry import TOOL
from .types import ToolResult

__all__ = ["ToolSpec", "ToolRegistry", "tool", "parse_tool_calls", "default_registry"]


# --------------------------------------------------------------------------- #
# 类型 → JSON Schema
# --------------------------------------------------------------------------- #
def _json_type(py_type: Any) -> Dict[str, Any]:
    origin = get_origin(py_type)
    if origin is typing.Union:                       # Optional[X]
        args = [a for a in get_args(py_type) if a is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
        return {"anyOf": [_json_type(a) for a in args]}
    if origin in (list, typing.List):
        item = get_args(py_type)[0] if get_args(py_type) else Any
        return {"type": "array", "items": _json_type(item)}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean",
               bytes: "string", type(None): "null"}
    if py_type in mapping:
        return {"type": mapping[py_type]}
    return {}


def _parse_docstring(doc: Optional[str]) -> tuple[str, Dict[str, str]]:
    """从 docstring 里提取概述与 ``:param x:`` 说明。"""
    if not doc:
        return "", {}
    lines = [l.strip() for l in doc.strip().splitlines()]
    desc_lines, params = [], {}
    for l in lines:
        if l.startswith(":param "):
            rest = l[len(":param "):]
            if ":" in rest:
                k, v = rest.split(":", 1)
                params[k.strip()] = v.strip()
        elif l.startswith(":return"):
            break
        else:
            desc_lines.append(l)
    return " ".join(desc_lines).strip(), params


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable] = None
    is_async: bool = False
    dangerous: bool = False            # 需要沙箱 / 二次确认
    requires_confirmation: bool = False

    def schema(self) -> Dict[str, Any]:
        """OpenAI function-calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    # ------------------------------------------------------------------ #
    def invoke(self, arguments: Dict[str, Any], call_id: str = "") -> ToolResult:
        t0 = time.time()
        try:
            kwargs = self._coerce(arguments)
            out = self.func(**kwargs) if self.func else ""
            if inspect.isawaitable(out):
                out = asyncio.get_event_loop().run_until_complete(out)
            return ToolResult(call_id=call_id, name=self.name, content=str(out),
                              latency_ms=(time.time() - t0) * 1000)
        except Exception as exc:                       # 错误回灌给模型
            return ToolResult(call_id=call_id, name=self.name,
                              content=f"工具 {self.name} 执行失败: {type(exc).__name__}: {exc}",
                              is_error=True, latency_ms=(time.time() - t0) * 1000)

    def _coerce(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """按类型标注做一次轻量校验/转换（避免 LLM 把 "3" 当字符串传进来）。"""
        if self.func is None:
            return arguments
        hints = get_type_hints(self.func)
        sig = inspect.signature(self.func)
        out: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in arguments:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"缺少参数: {name}")
                continue
            v = arguments[name]
            t = hints.get(name, Any)
            try:
                out[name] = _coerce_value(v, t)
            except Exception:
                out[name] = v
        return out


def _coerce_value(v: Any, t: Any) -> Any:
    origin = get_origin(t)
    if origin is typing.Union:
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return _coerce_value(v, args[0])
        return v
    if t is int and isinstance(v, str):
        return int(v)
    if t is float and isinstance(v, str):
        return float(v)
    if t is bool and isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    if origin in (list, typing.List) and isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return v


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
class ToolRegistry:
    def __init__(self) -> None:
        self.specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self.specs:
            raise ValueError(f"工具重名: {spec.name}")
        self.specs[spec.name] = spec
        TOOL._items.setdefault(spec.name, spec.func)
        return spec

    def register_fn(self, func: Callable, name: Optional[str] = None,
                    description: Optional[str] = None, dangerous: bool = False) -> ToolSpec:
        doc_desc, param_docs = _parse_docstring(func.__doc__)
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, p in sig.parameters.items():
            if pname in ("self",):
                continue
            t = hints.get(pname, str)
            prop = _json_type(t)
            if pname in param_docs:
                prop["description"] = param_docs[pname]
            props[pname] = prop
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        schema = {"type": "object", "properties": props, "required": required}
        spec = ToolSpec(
            name=name or func.__name__,
            description=description or doc_desc or func.__name__,
            parameters=schema, func=func,
            is_async=inspect.iscoroutinefunction(func), dangerous=dangerous,
        )
        return self.register(spec)

    def add(self, func_or_spec):
        """既支持传函数，也支持传 ToolSpec。"""
        if isinstance(func_or_spec, ToolSpec):
            return self.register(func_or_spec)
        return self.register_fn(func_or_spec)

    def get(self, name: str) -> ToolSpec:
        if name not in self.specs:
            raise KeyError(f"未注册的工具: {name}；可用: {sorted(self.specs)}")
        return self.specs[name]

    def schemas(self) -> List[Dict[str, Any]]:
        return [s.schema() for s in self.specs.values()]

    def describe(self) -> str:
        lines = []
        for s in self.specs.values():
            req = s.parameters.get("required", [])
            params = ", ".join(f"{k}: {v.get('type', 'any')}"
                               f"{'' if k not in req else ''}"
                               for k, v in s.parameters.get("properties", {}).items())
            lines.append(f"- {s.name}({params}) : {s.description}")
        return "\n".join(lines)

    def invoke(self, name: str, arguments: Dict[str, Any], call_id: str = "") -> ToolResult:
        return self.get(name).invoke(arguments, call_id)

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __len__(self) -> int:
        return len(self.specs)


def tool(name: Optional[str] = None, description: Optional[str] = None,
         dangerous: bool = False, registry: Optional[ToolRegistry] = None):
    """装饰器：把函数注册为工具。"""
    def deco(func: Callable) -> Callable:
        (registry or default_registry).register_fn(func, name, description, dangerous)
        return func
    return deco


default_registry = ToolRegistry()


# --------------------------------------------------------------------------- #
# 从模型输出里解析工具调用
# --------------------------------------------------------------------------- #
_TAG_RE = None


def parse_tool_calls(text: str) -> List[Any]:
    """解析两种常见格式（OpenAI 风格 JSON，以及 ``<tool_call>...</tool_call>``）。

    小模型往往吐不出严格 JSON，因此这里做**尽力解析**：
    支持裸 JSON 对象、```json 代码块、以及标签包裹三种形态。
    """
    import json
    import re

    from .types import ToolCall

    calls: List[ToolCall] = []
    if not text:
        return calls

    blocks: List[str] = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        blocks.append(m.group(1).strip())
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        blocks.append(m.group(1).strip())
    if not blocks:
        blocks = [text.strip()]

    for b in blocks:
        try:
            obj = json.loads(b)
        except Exception:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = it.get("name") or it.get("tool") or it.get("function")
            args = it.get("arguments") or it.get("args") or it.get("parameters") or {}
            if not isinstance(args, dict):
                args = {"input": args}
            if nm:
                calls.append(ToolCall(name=str(nm), arguments=args))
    return calls
