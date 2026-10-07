"""上下文管理：把"无限增长的对话"塞进"有限的上下文窗口"。

这是长任务 Agent 的**第一工程难题**。常见策略组合：
  1. **滑动窗口**：只保留最近 N 条（简单，但会丢掉早期的关键约束）；
  2. **摘要压缩**：把早期内容用 LLM 压成一段（有损但保语义）；
  3. **检索注入**：只把"与当前问题相关"的历史片段取回来（RAG 思路）；
  4. **工具结果截断**：工具返回几千 token 时先截断再入库；
  5. **系统提示常驻**：系统约束永不被压缩掉。

这里 1+2+4+5 都实现了，3 由 ``memory.VectorMemory`` 提供。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .types import Message, Role

__all__ = ["ContextConfig", "ContextManager", "count_tokens"]


def count_tokens(text: str, tokenizer=None) -> int:
    """token 计数：有分词器就精确算，否则用 ~4 字符/token 的经验估计。"""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(len(text) // 4, 1)


@dataclass
class ContextConfig:
    max_context_tokens: int = 2048
    reserve_for_output: int = 256
    compaction_ratio: float = 0.85      # 用到多少比例就触发压缩
    keep_recent: int = 6                # 压缩时无条件保留的最近条数
    max_tool_result_chars: int = 2000   # 工具返回值截断
    always_keep_system: bool = True


class ContextManager:
    def __init__(self, cfg: Optional[ContextConfig] = None, tokenizer=None,
                 summarize_fn: Optional[Callable[[str], str]] = None) -> None:
        self.cfg = cfg or ContextConfig()
        self.tokenizer = tokenizer
        self.summarize_fn = summarize_fn
        self.stats = {"compactions": 0, "truncated_results": 0, "dropped": 0}

    # ------------------------------------------------------------------ #
    def budget(self) -> int:
        return max(self.cfg.max_context_tokens - self.cfg.reserve_for_output, 128)

    def size(self, messages: List[Message]) -> int:
        return sum(count_tokens(m.content, self.tokenizer) for m in messages)

    def should_compact(self, messages: List[Message]) -> bool:
        return self.size(messages) > self.budget() * self.cfg.compaction_ratio

    # ------------------------------------------------------------------ #
    def truncate_tool_result(self, content: str) -> str:
        if len(content) <= self.cfg.max_tool_result_chars:
            return content
        self.stats["truncated_results"] += 1
        keep = self.cfg.max_tool_result_chars
        head = keep // 2
        tail = keep - head
        return content[:head] + f"\n...[截断 {len(content) - keep} 字符]...\n" + content[-tail:]

    # ------------------------------------------------------------------ #
    def compact(self, messages: List[Message]) -> List[Message]:
        """把中间部分压成一条摘要，保留 system 与最近 keep_recent 条。"""
        self.stats["compactions"] += 1
        system = [m for m in messages if m.role == Role.SYSTEM] if self.cfg.always_keep_system else []
        rest = [m for m in messages if m not in system]
        keep = self.cfg.keep_recent
        if len(rest) <= keep:
            return messages
        old, recent = rest[:-keep], rest[-keep:]
        blob = "\n".join(f"{m.role.value}: {m.content[:600]}" for m in old)
        if self.summarize_fn is not None:
            try:
                blob = self.summarize_fn(blob)
            except Exception:
                blob = blob[:1500]
        else:
            blob = blob[:1500]
        summary = Message(role=Role.SYSTEM,
                          content=f"[历史摘要]\n{blob}",
                          metadata={"kind": "compaction"})
        out = system + [summary] + recent
        self.stats["dropped"] += len(old)
        return out

    # ------------------------------------------------------------------ #
    def build(self, system_prompt: str, memory_text: str,
              history: List[Message], retrieved: Optional[List[str]] = None) -> List[Message]:
        msgs: List[Message] = []
        if system_prompt:
            msgs.append(Message(role=Role.SYSTEM, content=system_prompt))
        if memory_text:
            msgs.append(Message(role=Role.SYSTEM, content=f"[记忆]\n{memory_text}"))
        if retrieved:
            body = "\n---\n".join(retrieved)
            msgs.append(Message(role=Role.SYSTEM, content=f"[检索到的相关记忆]\n{body}"))
        msgs.extend(history)
        if self.should_compact(msgs):
            msgs = self.compact(msgs)
        return msgs
