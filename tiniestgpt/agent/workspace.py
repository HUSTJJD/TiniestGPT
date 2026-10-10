"""Context Engineering：给 Agent 一个**受控的计算环境**。

2026 年长程 Agent 的结构已经更接近一个受控环境，而不是"一个会调工具的对话"：

    独立文件系统 + Shell/代码执行 + 持久化任务状态 + 记忆与技能
    + 测试工具 + 权限系统 + **可恢复的执行机制**

关键认知：**上下文工程取代了提示词优化**。
不是把 prompt 写得更巧，而是把"模型能看到什么"做成一套可管理的工程对象。

本模块提供两块：

* :class:`Workspace` —— 持久化工作区（文件 + 任务状态 + 执行历史），
  支持**断点续跑**：把状态落盘，换进程也能接着跑；
* :class:`SkillLibrary` —— 技能库：从成功轨迹里沉淀可复用的"技能"，
  失败轨迹则记录成教训（自我改进闭环的第一步）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Workspace", "TaskState", "Skill", "SkillLibrary"]


@dataclass
class TaskState:
    name: str
    step: int = 0
    done: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class Workspace:
    """持久化工作区：文件 + 状态 + 历史，全部落盘。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "files").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    # ---------------- 文件 ---------------- #
    def _p(self, name: str) -> Path:
        # 防路径穿越：任何 ../ 都被折叠掉
        return (self.root / "files" / Path(name).name).resolve()

    def write(self, name: str, content: str) -> str:
        p = self._p(name)
        p.write_text(content, encoding="utf-8")
        return str(p)

    def read(self, name: str) -> str:
        p = self._p(name)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def ls(self) -> List[str]:
        return sorted(x.name for x in (self.root / "files").iterdir())

    # ---------------- 状态（可恢复的关键） ---------------- #
    def save_state(self, st: TaskState) -> None:
        st.updated_at = time.time()
        (self.root / "state" / f"{st.name}.json").write_text(
            json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_state(self, name: str) -> Optional[TaskState]:
        p = self.root / "state" / f"{name}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return TaskState(**d)

    def log(self, event: str, **kw) -> None:
        self.history.append({"t": time.time(), "event": event, **kw})
        (self.root / "history.jsonl").open("a", encoding="utf-8").write(
            json.dumps(self.history[-1], ensure_ascii=False) + "\n")

    def resume(self, name: str) -> TaskState:
        """断点续跑：有存档就接着跑，没有就新建。"""
        return self.load_state(name) or TaskState(name=name)

    def report(self) -> str:
        return (f"Workspace({self.root}): {len(self.ls())} 个文件，"
                f"{len(list((self.root / 'state').glob('*.json')))} 份状态，"
                f"{len(self.history)} 条历史")


# --------------------------------------------------------------------------- #
@dataclass
class Skill:
    """一个可复用的技能：名字 + 触发条件 + 具体做法 + 成功率统计。"""

    name: str
    when: str                      # 什么时候该用
    how: str                       # 怎么做（可以是一段代码/一串步骤）
    uses: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.uses, 1)

    def as_prompt(self) -> str:
        return f"[技能] {self.name}\n适用：{self.when}\n做法：{self.how}"


class SkillLibrary:
    """技能库：从轨迹里沉淀，按语义相似度检索。

    检索用**字符 2-gram 的 Jaccard 相似度**——不依赖任何 embedding，
    离线可跑、行为可预测；真实系统可以换成向量检索。
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self.skills: Dict[str, Skill] = {}
        self.lessons: List[str] = []
        if self.path and self.path.exists():
            self._load()

    # ---------------- 沉淀 ---------------- #
    def learn(self, name: str, when: str, how: str) -> Skill:
        s = self.skills.get(name)
        if s is None:
            s = Skill(name=name, when=when, how=how)
            self.skills[name] = s
        else:
            s.how = how
        self._save()
        return s

    def record_use(self, name: str, success: bool) -> None:
        s = self.skills.get(name)
        if s is None:
            return
        s.uses += 1
        s.successes += int(success)
        self._save()

    def add_lesson(self, text: str) -> None:
        """失败轨迹沉淀成"教训"——避免同一个坑踩两次。"""
        if text and text not in self.lessons:
            self.lessons.append(text)
            self._save()

    # ---------------- 检索 ---------------- #
    @staticmethod
    def _sim(a: str, b: str) -> float:
        ga = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
        gb = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
        return len(ga & gb) / max(len(ga | gb), 1)

    def retrieve(self, query: str, k: int = 2) -> List[Skill]:
        scored = sorted(self.skills.values(),
                        key=lambda s: self._sim(query, s.when + s.name), reverse=True)
        return [s for s in scored[:k] if self._sim(query, s.when + s.name) > 0.0][:k]

    def as_context(self, query: str, k: int = 2) -> str:
        hits = self.retrieve(query, k)
        if not hits:
            return ""
        return "\n".join(s.as_prompt() for s in hits) + \
               (("\n教训：\n" + "\n".join("- " + x for x in self.lessons[-3:]))
                if self.lessons else "")

    # ---------------- 持久化 ---------------- #
    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"skills": {n: asdict(s) for n, s in self.skills.items()},
             "lessons": self.lessons}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.skills = {n: Skill(**s) for n, s in d.get("skills", {}).items()}
        self.lessons = d.get("lessons", [])

    def report(self) -> str:
        return (f"SkillLibrary: {len(self.skills)} 个技能，{len(self.lessons)} 条教训，"
                f"平均成功率 "
                f"{sum(s.success_rate for s in self.skills.values()) / max(len(self.skills), 1):.1%}")
