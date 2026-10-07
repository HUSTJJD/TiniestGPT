"""可观测性：Trace 与 Metrics。

Agent 的失败往往"不可复现"——因为 LLM 有随机性、工具会超时、上下文会漂移。
因此必须把**每一次运行完整落盘**（事件流 + 输入 + 输出），
才能离线重放、定位问题。这里用 JSONL，一行一个事件，便于 grep 与流式分析。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Event, EventType

__all__ = ["Tracer", "Metrics"]


class Tracer:
    """事件流记录器（可选写 JSONL 文件）。"""

    def __init__(self, path: Optional[str | Path] = None, run_id: str = "",
                 keep_in_memory: int = 2000) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.events: List[Event] = []
        self._lock = threading.Lock()
        self._keep = keep_in_memory
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def emit(self, type: EventType, step: int = 0, **payload: Any) -> Event:
        ev = Event(type=type, payload=payload, run_id=self.run_id, step=step)
        with self._lock:
            self.events.append(ev)
            if len(self.events) > self._keep:
                self.events.pop(0)
            if self._fh:
                self._fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
                self._fh.flush()
        return ev

    def filter(self, type: EventType) -> List[Event]:
        return [e for e in self.events if e.type == type]

    def tool_calls(self) -> List[Event]:
        return self.filter(EventType.TOOL_CALL)

    def errors(self) -> List[Event]:
        return self.filter(EventType.ERROR)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Metrics:
    """轻量计数器 / 计时器聚合。"""

    def __init__(self) -> None:
        self.counters: Dict[str, int] = {}
        self.times: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def record_time(self, name: str, ms: float) -> None:
        with self._lock:
            self.times.setdefault(name, []).append(ms)

    def avg(self, name: str) -> float:
        xs = self.times.get(name, [])
        return sum(xs) / len(xs) if xs else 0.0

    def p95(self, name: str) -> float:
        xs = sorted(self.times.get(name, []))
        if not xs:
            return 0.0
        return xs[int(0.95 * (len(xs) - 1))]

    def report(self) -> str:
        lines = [f"{k}: {v}" for k, v in sorted(self.counters.items())]
        for k in sorted(self.times):
            lines.append(f"{k}.avg: {self.avg(k):.1f}ms (p95 {self.p95(k):.1f}ms)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"counters": dict(self.counters),
                "times": {k: {"avg": self.avg(k), "p95": self.p95(k)} for k in self.times}}
