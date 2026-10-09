"""内置评测任务：全部**离线合成**，不依赖任何下载。

设计原则：任务必须"小模型也能做对一部分"，否则分数全是 0 就没有度量意义。
因此这里刻意选了 25M 参数模型也能学到东西的四类任务：

1. :data:`ARITHMETIC`  —— 两位数的加减法（可验证，**RLVR 的天然靶子**）
2. :data:`NEEDLE`      —— 长上下文里找一个数字（长程召回）
3. :data:`JSON_FORMAT` —— 按 schema 输出（工具调用 / 结构化输出的前置能力）
4. :data:`CLOZE`       —— 完形填空，用 log-likelihood 打分（不依赖生成质量）

前三类用 **生成 + 规则判定**（这也是 2026 年 RLVR 的标准打分方式：
答案对不对由规则/解析器说了算，不需要另一个模型来评判）。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["EvalTask", "TASKS", "build_task", "available_tasks"]


# --------------------------------------------------------------------------- #
#  打分器（全部是"可验证"的规则，不引入第二个模型）
# --------------------------------------------------------------------------- #
def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def score_arithmetic(pred: str, ex: Dict[str, Any]) -> float:
    got = _extract_int(pred)
    return 1.0 if got is not None and got == ex["answer"] else 0.0


def score_needle(pred: str, ex: Dict[str, Any]) -> float:
    got = _extract_int(pred)
    return 1.0 if got is not None and got == ex["answer"] else 0.0


def score_json(pred: str, ex: Dict[str, Any]) -> float:
    """能解析出 JSON 且必填字段齐全/类型正确才给分（部分给 0.5）。"""
    start, end = pred.find("{"), pred.rfind("}")
    if start < 0 or end <= start:
        return 0.0
    try:
        obj = json.loads(pred[start:end + 1])
    except Exception:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    need: Dict[str, str] = ex["answer"]
    hit = sum(1 for k, t in need.items()
              if k in obj and isinstance(obj[k], {"str": str, "int": int, "num": (int, float)}[t]))
    return hit / len(need)


def score_cloze(pred: str, ex: Dict[str, Any]) -> float:
    return 1.0 if pred.strip().lower().startswith(str(ex["answer"]).lower()) else 0.0


# --------------------------------------------------------------------------- #
@dataclass
class EvalTask:
    name: str
    kind: str                                   # "generate" | "loglikelihood"
    description: str
    scorer: Callable[[str, Dict[str, Any]], float]
    n_examples: int = 32
    max_new_tokens: int = 32
    stop: List[str] = field(default_factory=lambda: ["\n"])

    def build(self, n: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
        n = n or self.n_examples
        rng = random.Random(hash((self.name, seed)) & 0xFFFFFFFF)
        return self._build(n, rng)

    def _build(self, n: int, rng: random.Random) -> List[Dict[str, Any]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class ArithmeticTask(EvalTask):
    def _build(self, n, rng):
        out = []
        for _ in range(n):
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            op = rng.choice(["+", "-"])
            ans = a + b if op == "+" else a - b
            out.append({"prompt": f"Calculate: {a} {op} {b} =\nAnswer:",
                        "answer": ans, "meta": f"{a}{op}{b}"})
        return out


class NeedleTask(EvalTask):
    def _build(self, n, rng):
        filler = ("The little girl walked through the garden and saw a red flower. "
                  "She smiled and kept walking. ")
        out = []
        for _ in range(n):
            key = rng.randint(1000, 9999)
            k = rng.randint(3, 6)
            body = filler * k
            out.append({"prompt": (f"Remember this number: {key}.\n{body}"
                                   f"What is the number?\nAnswer:"),
                        "answer": key, "meta": f"needle={key},paras={k}"})
        return out


class JsonFormatTask(EvalTask):
    def _build(self, n, rng):
        names = ["Alice", "Bob", "Cleo", "Dan", "Eve", "Fay"]
        out = []
        for _ in range(n):
            nm = rng.choice(names)
            age = rng.randint(5, 80)
            out.append({
                "prompt": (f'Output a JSON object for a person named {nm} aged {age}.\n'
                           f'Use keys "name" (string) and "age" (int).\nJSON:'),
                "answer": {"name": "str", "age": "int"}, "meta": f"{nm}/{age}",
            })
        return out


class ClozeTask(EvalTask):
    """完形填空：用 log-likelihood 打分，绕开"小模型生成能力弱"的干扰。"""

    def _build(self, n, rng):
        bank = [
            ("The cat sat on the", ["mat", "car", "cloud", "mountain"], "mat"),
            ("She drank a cup of hot", ["tea", "stone", "bicycle", "yesterday"], "tea"),
            ("The sun rises in the", ["east", "south", "basement", "kitchen"], "east"),
            ("A dog says", ["woof", "moo", "meow", "beep"], "woof"),
            ("Birds fly in the", ["sky", "ocean", "cave", "drawer"], "sky"),
            ("He opened the", ["door", "idea", "silence", "Tuesday"], "door"),
            ("Water is", ["wet", "loud", "square", "invisible"], "wet"),
            ("The opposite of hot is", ["cold", "fast", "loud", "green"], "cold"),
        ]
        out = []
        for i in range(n):
            stem, opts, gold = bank[i % len(bank)]
            opts = list(opts)
            rng.shuffle(opts)
            out.append({"prompt": f"{stem} ", "choices": opts, "answer": gold,
                        "meta": stem})
        return out


# --------------------------------------------------------------------------- #
TASKS: Dict[str, EvalTask] = {
    "arithmetic": ArithmeticTask(
        name="arithmetic", kind="generate", scorer=score_arithmetic,
        description="两位数加减法（可验证奖励的靶子任务）", n_examples=32, max_new_tokens=8),
    "needle": NeedleTask(
        name="needle", kind="generate", scorer=score_needle,
        description="长上下文数字召回（needle-in-a-haystack 迷你版）",
        n_examples=16, max_new_tokens=8),
    "json": JsonFormatTask(
        name="json", kind="generate", scorer=score_json,
        description="按 schema 输出 JSON（工具调用前置能力）",
        n_examples=16, max_new_tokens=48),
    "cloze": ClozeTask(
        name="cloze", kind="loglikelihood", scorer=score_cloze,
        description="完形填空（log-likelihood 打分，不依赖生成）", n_examples=32),
}


def available_tasks() -> List[str]:
    return sorted(TASKS)


def build_task(name: str, n: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    if name not in TASKS:
        raise KeyError(f"未知评测任务: {name}；可用: {available_tasks()}")
    return TASKS[name].build(n, seed)
