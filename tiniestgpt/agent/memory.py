"""分层记忆：工作记忆 / 摘要记忆 / 向量记忆 / 情景记忆。

对应认知科学的分层，也对应工程上的取舍：

* **工作记忆**（WorkingMemory）：当前对话的原始消息，精确但昂贵（占 context）；
* **摘要记忆**（SummaryMemory）：把滚出窗口的内容压成一段话，便宜但有损；
* **向量记忆**（VectorMemory）：语义检索，容量大，用于"长期知识"；
* **情景记忆**（EpisodicMemory）：记住"做过什么、结果如何"，
  让 Agent 在遇到相似任务时能复用过去的成功经验（Reflexion 的核心）。

向量检索这里用**哈希 + TF-IDF 风格权重**实现（零外部依赖、可离线跑），
同时保留 ``embed_fn`` 接口，方便换成真正的句向量模型。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["MemoryItem", "WorkingMemory", "SummaryMemory", "VectorMemory",
           "EpisodicMemory", "MemoryManager", "hashing_embed"]


@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    score: float = 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def hashing_embed(text: str, dim: int = 256) -> np.ndarray:
    """特征哈希（hashing trick）：把词映射到固定维度再加权，无需训练、无需外部模型。"""
    vec = np.zeros(dim, dtype=np.float32)
    toks = _tokenize(text)
    if not toks:
        return vec
    tf: Dict[str, float] = {}
    for t in toks:
        tf[t] = tf.get(t, 0.0) + 1.0
    for t, c in tf.items():
        h = int(hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0     # 有符号哈希，降低碰撞偏差
        vec[idx] += sign * (1.0 + math.log(c))
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


class WorkingMemory:
    """有界队列：超出容量时把最老的挤出去（返回给上层做摘要）。"""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, text: str, **meta) -> MemoryItem:
        it = MemoryItem(text=text, meta=meta)
        self.items.append(it)
        return it

    def pop_overflow(self) -> List[MemoryItem]:
        if len(self.items) <= self.capacity:
            return []
        n = len(self.items) - self.capacity
        out = self.items[:n]
        self.items = self.items[n:]
        return out

    def recent(self, k: int = 8) -> List[MemoryItem]:
        return self.items[-k:]

    def clear(self) -> None:
        self.items.clear()


class SummaryMemory:
    """把溢出的工作记忆压成一条摘要（可用 LLM，也可降级为抽取式）。"""

    def __init__(self, compress_fn: Optional[Callable[[str], str]] = None,
                 max_items: int = 50) -> None:
        self.compress_fn = compress_fn
        self.items: List[MemoryItem] = []
        self.max_items = max_items

    def absorb(self, items: List[MemoryItem]) -> None:
        if not items:
            return
        blob = "\n".join(i.text[:400] for i in items)
        if self.compress_fn is not None:
            try:
                blob = self.compress_fn(blob)
            except Exception:
                pass
        else:
            blob = blob[:1200] + ("..." if len(blob) > 1200 else "")
        self.items.append(MemoryItem(text=blob, meta={"kind": "summary", "n_src": len(items)}))
        if len(self.items) > self.max_items:
            # 老摘要再合并一次，避免无限增长（层次化摘要）
            merged = "\n".join(i.text for i in self.items[:2])
            self.items = [MemoryItem(text=merged[:1500], meta={"kind": "summary"})] + self.items[2:]

    def text(self) -> str:
        return "\n".join(f"[摘要] {i.text}" for i in self.items[-3:])

    def clear(self) -> None:
        self.items.clear()


class VectorMemory:
    """语义检索记忆（余弦相似度 + 可选重排）。"""

    def __init__(self, dim: int = 256, embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> None:
        self.dim = dim
        self.embed_fn = embed_fn or (lambda t: hashing_embed(t, dim))
        self.items: List[MemoryItem] = []
        self.matrix: Optional[np.ndarray] = None

    def add(self, text: str, **meta) -> MemoryItem:
        it = MemoryItem(text=text, meta=meta)
        self.items.append(it)
        v = self.embed_fn(text).reshape(1, -1)
        self.matrix = v if self.matrix is None else np.vstack([self.matrix, v])
        return it

    def search(self, query: str, k: int = 3) -> List[MemoryItem]:
        if self.matrix is None or not self.items:
            return []
        q = self.embed_fn(query).reshape(1, -1)
        sims = (self.matrix @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            it = self.items[int(i)]
            it.score = float(sims[int(i)])
            out.append(it)
        return out

    def __len__(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items.clear()
        self.matrix = None


class EpisodicMemory(VectorMemory):
    """情景记忆：存"任务 → 轨迹 → 结果"，用于经验复用与自我反思。"""

    def record(self, task: str, trajectory: str, success: bool, score: float = 0.0) -> MemoryItem:
        return self.add(
            text=f"任务: {task}\n做法: {trajectory[:600]}\n结果: {'成功' if success else '失败'}",
            meta={"success": success, "score": score, "kind": "episode"},
        )

    def similar_successes(self, task: str, k: int = 2) -> List[MemoryItem]:
        hits = self.search(task, k=max(k * 3, 6))
        return [h for h in hits if h.meta.get("success")][:k]


class MemoryManager:
    """把四层记忆组织成一个统一入口。"""

    def __init__(self, working_capacity: int = 24, dim: int = 256,
                 compress_fn: Optional[Callable[[str], str]] = None) -> None:
        self.working = WorkingMemory(working_capacity)
        self.summary = SummaryMemory(compress_fn)
        self.semantic = VectorMemory(dim)
        self.episodic = EpisodicMemory(dim)

    def observe(self, text: str, **meta) -> None:
        self.working.add(text, **meta)
        self.summary.absorb(self.working.pop_overflow())

    def remember(self, text: str, **meta) -> None:
        """长期知识（写入向量记忆）。"""
        self.semantic.add(text, **meta)

    def recall(self, query: str, k: int = 3) -> List[MemoryItem]:
        return self.semantic.search(query, k)

    def context_text(self) -> str:
        parts = []
        s = self.summary.text()
        if s:
            parts.append(s)
        w = "\n".join(i.text for i in self.working.recent(8))
        if w:
            parts.append(w)
        return "\n".join(parts)

    def clear(self) -> None:
        self.working.clear()
        self.summary.clear()
        self.semantic.clear()
        self.episodic.clear()
