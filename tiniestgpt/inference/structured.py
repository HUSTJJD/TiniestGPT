"""结构化输出 / 语法约束解码（logits masking）。

为什么需要它：
让模型"输出一个 JSON"最朴素的做法是在 prompt 里写"请输出 JSON"，
然后把结果丢给 ``json.loads`` —— 一旦模型多说一句"好的，这是你要的："，解析就崩了。
Agent 框架里这会直接让工具调用失败，所以生产引擎（vLLM / SGLang / Outlines）
都提供 **约束解码**：**在每一步把不合法的 token 的 logits 置为 -inf**，
让模型在语法上"不可能走错"。

原理其实只有两步：

1. 维护一个**增量语法状态机**（这里是一个 JSON 前缀校验器），
   它知道"当前位置上哪些字符是合法的"；
2. 每一步把词表里所有 token 逐个喂给状态机的**副本**，
   走不通的就 mask 掉。

代价：每步要扫一遍词表（本项目词表几千，可以接受；
生产实现会用预编译的 DFA + 前缀树把这一步压到微秒级）。

用法::

    from tiniestgpt.inference.structured import StructuredDecoder
    dec = StructuredDecoder(tokenizer, vocab_size, schema={"type": "object"})
    logits = dec.mask_logits(logits)        # 把非法 token 置为 -inf
    dec.feed_token(token_id)                # 采样后推进状态
    dec.complete                            # JSON 是否已经闭合
"""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["JsonPrefix", "StructuredDecoder", "validate_against_schema"]


# --------------------------------------------------------------------------- #
class JsonPrefix:
    """JSON **前缀**校验器：逐字符喂入，判断"到目前为止是否还可能成为合法 JSON"。"""

    __slots__ = ("stack", "state", "in_str", "escape", "str_role", "lit", "num", "done")

    def __init__(self) -> None:
        self.stack: List[str] = []      # 'o' = object, 'a' = array
        self.state = "value"            # value | key | colon | comma | number | literal | end
        self.in_str = False
        self.escape = False
        self.str_role = ""              # 'key' | 'value'
        self.lit = ""
        self.num = ""
        self.done = False

    # ------------------------------------------------------------------ #
    def clone(self) -> "JsonPrefix":
        other = JsonPrefix()
        other.stack = list(self.stack)
        other.state = self.state
        other.in_str = self.in_str
        other.escape = self.escape
        other.str_role = self.str_role
        other.lit = self.lit
        other.num = self.num
        other.done = self.done
        return other

    def key(self) -> Tuple:
        return (tuple(self.stack), self.state, self.in_str, self.escape,
                self.str_role, self.lit, self.num, self.done)

    # ------------------------------------------------------------------ #
    def _close(self, c: str) -> bool:
        if c == "}" and self.stack and self.stack[-1] == "o":
            self.stack.pop()
        elif c == "]" and self.stack and self.stack[-1] == "a":
            self.stack.pop()
        else:
            return False
        if self.stack:
            self.state = "comma"
        else:
            self.state = "end"
            self.done = True
        return True

    def feed(self, c: str) -> bool:
        """喂入一个字符；返回 False 表示"这条路已经不可能合法了"。"""
        if self.done or self.state == "end":
            return c in " \t\n\r"

        if self.in_str:
            if self.escape:
                self.escape = False
                return True
            if c == "\\":
                self.escape = True
                return True
            if c == '"':
                self.in_str = False
                if self.str_role == "key":
                    self.state = "colon"
                else:
                    self.state = "comma"
                return True
            return True                                  # 字符串内容：任意字符

        if c in " \t\n\r":
            return True

        if self.state == "key":
            if c == '"':
                self.str_role, self.in_str = "key", True
                return True
            return self._close(c) if c in "}" else False

        if self.state == "colon":
            if c == ":":
                self.state = "value"
                return True
            return False

        if self.state == "comma":
            if c == ",":
                self.state = "key" if (self.stack and self.stack[-1] == "o") else "value"
                return True
            return self._close(c)

        if self.state == "value":
            if c == "{":
                self.stack.append("o")
                self.state = "key"
                return True
            if c == "[":
                self.stack.append("a")
                self.state = "value"
                return True
            if c == '"':
                self.str_role, self.in_str = "value", True
                return True
            if c == "-" or c.isdigit():
                self.num, self.state = c, "number"
                return True
            if c in "tfn":
                self.lit, self.state = c, "literal"
                return True
            return False

        if self.state == "number":
            if c.isdigit() or c in ".eE+-":
                self.num += c
                return True
            self.state = "comma"                          # 数字结束
            return self.feed(c) if c not in " \t\n\r" else True

        if self.state == "literal":
            self.lit += c
            if self.lit in ("true", "false", "null"):
                self.state = "comma"
                return True
            return any(w.startswith(self.lit) for w in ("true", "false", "null"))

        return False

    def accepts(self, text: str) -> bool:
        """整段文本喂进去是否都合法（用副本，不污染自身）。"""
        probe = self.clone()
        return all(probe.feed(c) for c in text)


# --------------------------------------------------------------------------- #
class StructuredDecoder:
    """把一个请求约束成"必须输出合法 JSON"。

    :param tokenizer:   只要有 ``decode(List[int]) -> str``
    :param vocab_size:  词表大小
    :param schema:      可选的 JSON Schema（只做**事后校验**，约束本身是 JSON 语法级）
    """

    def __init__(self, tokenizer, vocab_size: int,
                 schema: Optional[dict] = None, start_with_object: bool = True) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.schema = schema
        self.state = JsonPrefix()
        self.text = ""
        if start_with_object:
            # 直接把开头的 '{' 吃掉：少一步采样，也避免模型先寒暄
            self.state.feed("{")
            self.text = "{"
        self._texts: List[str] = [self._decode(i) for i in range(self.vocab_size)]
        self._cache: Dict[Tuple, List[int]] = {}

    def _decode(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([token_id], skip_special=True)
        except TypeError:  # pragma: no cover - 不同分词器签名
            return self.tokenizer.decode([token_id])

    # ------------------------------------------------------------------ #
    def allowed_token_ids(self) -> List[int]:
        key = self.state.key()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        allowed = [i for i, t in enumerate(self._texts)
                   if t and self.state.accepts(t)]
        # 兜底：万一一个都不允许（分词器很怪），退化为允许全部，避免死循环
        if not allowed:
            allowed = list(range(self.vocab_size))
        if len(self._cache) < 512:
            self._cache[key] = allowed
        return allowed

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """把非法 token 置为 ``-inf``（保持形状与 dtype）。"""
        allowed = self.allowed_token_ids()
        if len(allowed) == self.vocab_size:
            return logits
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        mask[idx] = logits[idx]
        return mask

    def feed_token(self, token_id: int) -> None:
        text = self._texts[token_id] if 0 <= token_id < self.vocab_size else ""
        for c in text:
            self.state.feed(c)
        self.text += text

    # ------------------------------------------------------------------ #
    @property
    def complete(self) -> bool:
        return self.state.done

    def value(self):
        """解析当前文本；未闭合时返回 ``None``。"""
        try:
            return json.loads(self.text)
        except Exception:
            return None

    def validate(self) -> Tuple[bool, str]:
        """语法 + schema 双重校验。"""
        obj = self.value()
        if obj is None:
            return False, "JSON 尚未闭合或无法解析"
        if self.schema is None:
            return True, ""
        return validate_against_schema(obj, self.schema)


# --------------------------------------------------------------------------- #
def validate_against_schema(obj, schema: dict) -> Tuple[bool, str]:
    """极简 JSON Schema 校验（type / required / properties / enum / items）。"""
    t = schema.get("type")
    type_map = {"object": dict, "array": list, "string": str,
                "number": (int, float), "integer": int, "boolean": bool, "null": type(None)}
    if t and t in type_map:
        want = type_map[t]
        if t in ("number", "integer") and isinstance(obj, bool):
            return False, f"期望 {t}，得到 bool"
        if not isinstance(obj, want):
            return False, f"期望 {t}，得到 {type(obj).__name__}"
    if "enum" in schema and obj not in schema["enum"]:
        return False, f"{obj!r} 不在 enum {schema['enum']} 中"
    if isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                return False, f"缺少必填字段 {key!r}"
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj:
                ok, msg = validate_against_schema(obj[key], sub)
                if not ok:
                    return False, f"{key}: {msg}"
    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            ok, msg = validate_against_schema(item, schema["items"])
            if not ok:
                return False, f"[{i}]: {msg}"
    return True, ""
