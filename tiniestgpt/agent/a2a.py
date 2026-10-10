"""A2A（Agent-to-Agent）协议 + 图记忆。

**A2A**
MCP 解决"Agent ↔ 工具"的互操作，A2A 解决"Agent ↔ Agent"：
一个 Agent 把另一个 Agent 当成**服务**来调用，而不是共享内部状态。

最小协议只需要三件事：

* **Agent Card**：一个 JSON，描述"我是谁、我能干什么、怎么调我"；
* **Task**：一次调用是一个任务，有生命周期（submitted → working → completed/failed）；
* **Artifact**：任务产出（文本 / 文件 / 结构化数据）。

**图记忆**
四层记忆里的"情景记忆"用向量存，会丢失**关系**。
图记忆把 (主体, 关系, 客体) 三元组存下来，支持多跳查询——
这对"谁在什么时候说过什么"这类问题比向量检索准得多。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

__all__ = ["AgentCard", "A2ATask", "A2ARegistry", "GraphMemory"]


# --------------------------------------------------------------------------- #
@dataclass
class AgentCard:
    """A2A 的 Agent Card：能力声明 + 调用端点。"""

    name: str
    description: str = ""
    skills: List[str] = field(default_factory=list)
    endpoint: str = ""                 # 本实现里是本地回调名
    version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "description": self.description,
            "skills": self.skills, "endpoint": self.endpoint, "version": self.version,
        }, ensure_ascii=False)


@dataclass
class A2ATask:
    """一次 A2A 调用。状态机：submitted → working → completed / failed"""

    id: str
    sender: str
    receiver: str
    message: str
    state: str = "submitted"
    artifacts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition(self, state: str, artifact: str = "") -> None:
        self.state = state
        self.updated_at = time.time()
        if artifact:
            self.artifacts.append(artifact)

    def to_json(self) -> str:
        return json.dumps({"id": self.id, "sender": self.sender, "receiver": self.receiver,
                           "state": self.state, "artifacts": self.artifacts},
                          ensure_ascii=False)


class A2ARegistry:
    """本地 A2A 注册表：发卡片 → 按技能找 Agent → 发任务。"""

    def __init__(self) -> None:
        self.cards: Dict[str, AgentCard] = {}
        self.handlers: Dict[str, callable] = {}
        self.tasks: Dict[str, A2ATask] = {}
        self._seq = 0

    def register(self, card: AgentCard, handler: callable) -> None:
        self.cards[card.name] = card
        self.handlers[card.name] = handler

    def discover(self, skill: str) -> List[AgentCard]:
        """按技能发现 Agent（A2A 的核心价值：不用事先知道对方是谁）。"""
        return [c for c in self.cards.values() if (not skill) or (skill in c.skills)]

    def send(self, sender: str, receiver: str, message: str) -> A2ATask:
        self._seq += 1
        t = A2ATask(id=f"task-{self._seq}", sender=sender, receiver=receiver,
                    message=message)
        self.tasks[t.id] = t
        if receiver not in self.handlers:
            t.transition("failed", artifact=f"未知 Agent: {receiver}")
            return t
        t.transition("working")
        try:
            out = self.handlers[receiver](message)
            t.transition("completed", artifact=str(out))
        except Exception as exc:                       # noqa: BLE001
            t.transition("failed", artifact=f"{type(exc).__name__}: {exc}")
        return t

    def report(self) -> str:
        states: Dict[str, int] = {}
        for t in self.tasks.values():
            states[t.state] = states.get(t.state, 0) + 1
        return f"A2A: {len(self.cards)} 个 Agent，{len(self.tasks)} 个任务 {states}"


# --------------------------------------------------------------------------- #
class GraphMemory:
    """图记忆：存 (subject, relation, object) 三元组，支持多跳查询。"""

    def __init__(self) -> None:
        self.triples: List[Tuple[str, str, str]] = []
        self.index: Dict[str, Set[int]] = {}

    def _idx(self, key: str, i: int) -> None:
        self.index.setdefault(key, set()).add(i)

    def add(self, s: str, r: str, o: str) -> None:
        i = len(self.triples)
        self.triples.append((s, r, o))
        self._idx(f"s:{s}", i)
        self._idx(f"o:{o}", i)
        self._idx(f"r:{r}", i)

    def query(self, s: Optional[str] = None, r: Optional[str] = None,
              o: Optional[str] = None) -> List[Tuple[str, str, str]]:
        cand: Optional[Set[int]] = None
        for key, val in (("s", s), ("r", r), ("o", o)):
            if val is None:
                continue
            hit = self.index.get(f"{key}:{val}", set())
            cand = hit if cand is None else (cand & hit)
        if not cand:
            return []
        return [self.triples[i] for i in sorted(cand)]

    def multi_hop(self, start: str, hops: int = 2) -> List[Tuple[str, str, str]]:
        """从 start 出发走 hops 跳：这是向量检索做不到的事。"""
        seen: List[Tuple[str, str, str]] = []
        frontier = {start}
        for _ in range(max(hops, 1)):
            nxt: Set[str] = set()
            for node in frontier:
                for t in self.query(s=node):
                    if t not in seen:
                        seen.append(t)
                        nxt.add(t[2])
                for t in self.query(o=node):
                    if t not in seen:
                        seen.append(t)
                        nxt.add(t[0])
            frontier = nxt
            if not frontier:
                break
        return seen

    def report(self) -> str:
        rels = {r for _s, r, _o in self.triples}
        return f"GraphMemory: {len(self.triples)} 条三元组，{len(rels)} 种关系"
