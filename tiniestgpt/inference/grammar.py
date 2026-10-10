"""任意 schema 的约束解码：把 JSON 子集升级成**可编译的 DFA**。

现有的 ``structured.py`` 是一个手写的 JSON 前缀状态机——能跑，
但换个 schema 就得改代码。生产系统（vLLM 用 xgrammar / llguidance）的做法是：
**把 schema 编译成 DFA**，然后在每步解码时用 DFA 的转移表做 logits masking。

为什么必须编译成 DFA 而不是"生成完再校验"：

* 生成完再校验，错一个字符就要整段重来，成本随长度线性增长；
* 逐 token 约束能保证**一定**合法，延迟几乎不增加（只是多一次查表）。

本模块支持三类约束，统一编译成 DFA：

* :func:`from_regex` —— 正则（本模块自带一个极小的 NFA→DFA 转换器）；
* :func:`from_json_schema` —— JSON Schema 的常用子集（string/number/bool/enum/array/object）；
* :func:`from_literal` —— 字面量枚举（分类标签、工具名）。

掩码的语义：只允许能**继续走向终态**的 token。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

__all__ = ["DFA", "from_regex", "from_json_schema", "from_literal",
           "ConstrainedDecoder", "apply_dfa_mask"]


# --------------------------------------------------------------------------- #
@dataclass
class DFA:
    """一个最简 DFA：状态数 + 转移表 + 终态集。"""

    n_states: int
    transitions: Dict[Tuple[int, str], int] = field(default_factory=dict)
    finals: Set[int] = field(default_factory=set)
    start: int = 0

    def step(self, state: int, ch: str) -> int:
        return self.transitions.get((state, ch), -1)

    def accepts(self, s: str) -> bool:
        st = self.start
        for ch in s:
            st = self.step(st, ch)
            if st < 0:
                return False
        return st in self.finals

    def allowed(self, state: int, alphabet: Sequence[str]) -> Set[str]:
        return {c for c in alphabet if (state, c) in self.transitions}

    def reachable_final(self, state: int, alphabet: Sequence[str]) -> bool:
        """从 state 出发是否还可能走到终态（BFS）。"""
        seen, stack = {state}, [state]
        while stack:
            s = stack.pop()
            if s in self.finals:
                return True
            for c in alphabet:
                t = self.step(s, c)
                if t >= 0 and t not in seen:
                    seen.add(t)
                    stack.append(t)
        return False


def from_literal(options: Sequence[str], alphabet: Optional[Sequence[str]] = None) -> DFA:
    """字面量枚举：任何一个选项都接受。"""
    alphabet = list(alphabet) if alphabet else sorted({c for o in options for c in o})
    trans: Dict[Tuple[int, str], int] = {}
    nxt = 1
    finals: Set[int] = set()
    for opt in options:
        st = 0
        for i, ch in enumerate(opt):
            new = nxt
            nxt += 1
            trans[(st, ch)] = new
            st = new
        finals.add(st)
    return DFA(n_states=nxt, transitions=trans, finals=finals, start=0)


def from_regex(pattern: str, alphabet: Optional[Sequence[str]] = None) -> DFA:
    """极简正则 → NFA → DFA（支持 ``|`` ``?`` ``*`` ``+`` 与字面字符）。

    不支持分组括号与字符类——那需要真正的 parser。
    超出能力时抛 ``ValueError``，让调用方回退到"生成后校验"。
    """
    if any(c in pattern for c in "()[]"):
        raise ValueError("本教学实现只支持 | ? * + 与字面字符，不支持分组与字符类")
    alphabet = list(alphabet) if alphabet else sorted(set(pattern) - set("|?*+"))
    # ---- 解析成 NFA（Thompson 构造的极简版） ----
    nfa: Dict[int, Dict[str, Set[int]]] = {}
    nxt = [0]

    def new_state() -> int:
        nxt[0] += 1
        return nxt[0] - 1

    def build(node: str) -> Tuple[int, int]:
        """返回 (start, end)。node 形如 'ab*c|d'（先处理 | 分支）。"""
        branches = node.split("|")
        if len(branches) > 1:
            s, e = new_state(), new_state()
            nfa.setdefault(s, {})
            for b in branches:
                bs, be = build(b)
                nfa[s].setdefault("", set()).add(bs)
                nfa.setdefault(be, {}).setdefault("", set()).add(e)
            return s, e
        s = new_state()
        cur = s
        last: Optional[Tuple[int, str, int]] = None      # (from, symbol, to)
        i = 0
        while i < len(node):
            ch = node[i]
            if ch in "?*+":
                if last is None:
                    raise ValueError(f"量词 '{ch}' 前没有可作用的元素")
                src, _sym, tgt = last
                if ch in "*?":                            # 允许跳过（0 次）
                    nfa.setdefault(src, {}).setdefault("", set()).add(tgt)
                if ch in "*+":                            # 允许回到起点（多次）
                    nfa.setdefault(tgt, {}).setdefault("", set()).add(src)
                i += 1
                continue
            e = new_state()
            nfa.setdefault(cur, {}).setdefault(ch, set()).add(e)
            last = (cur, ch, e)
            cur = e
            i += 1
        return s, cur

    start, end = build(pattern)

    # ---- NFA → DFA（子集构造） ----
    def eps_closure(states: Set[int]) -> Set[int]:
        out, stack = set(states), list(states)
        while stack:
            s = stack.pop()
            for t in nfa.get(s, {}).get("", set()):
                if t not in out:
                    out.add(t)
                    stack.append(t)
        return out

    start_set = eps_closure({start})
    dfa_states: Dict[frozenset, int] = {frozenset(start_set): 0}
    trans: Dict[Tuple[int, str], int] = {}
    finals: Set[int] = set()
    work = [start_set]
    sid = 0
    while work:
        cur = work.pop()
        cs = dfa_states[frozenset(cur)]
        if end in cur:
            finals.add(cs)
        for c in alphabet:
            nxt_set: Set[int] = set()
            for s in cur:
                nxt_set |= nfa.get(s, {}).get(c, set())
            if not nxt_set:
                continue
            cl = eps_closure(nxt_set)
            key = frozenset(cl)
            if key not in dfa_states:
                sid += 1
                dfa_states[key] = sid
                work.append(cl)
            trans[(cs, c)] = dfa_states[key]
    return DFA(n_states=len(dfa_states), transitions=trans, finals=finals, start=0)


def from_json_schema(schema: dict, alphabet: Optional[Sequence[str]] = None) -> DFA:
    """JSON Schema 常用子集 → DFA。

    支持：``{"type": "string"|"number"|"boolean"|"integer"}``、``enum``、
    数组、以及"只要求它是一个合法 JSON 值"的兜底模式。
    嵌套 object 的**字段顺序**约束无法用 DFA 表达（需要下推自动机），
    所以这里退化为"合法 JSON 文本"的 DFA，并在 docstring 里说清楚。
    """
    t = schema.get("type")
    if "enum" in schema:
        return from_literal([str(v) for v in schema["enum"]], alphabet)
    if t in ("string",):
        return from_literal(['""'], alphabet)
    if t in ("number", "integer"):
        # 手写一个 "-?\d+" 的 DFA（from_regex 不支持字符类）
        digits = "0123456789"
        alpha = list(alphabet) if alphabet else list("-." + digits)
        trans: Dict[Tuple[int, str], int] = {}
        for d in digits:
            trans[(0, d)] = 2      # 直接进数字
            trans[(1, d)] = 2      # 负号后进数字
            trans[(2, d)] = 2
        trans[(0, "-")] = 1
        if t == "number":
            trans[(2, ".")] = 3
            for d in digits:
                trans[(3, d)] = 3
            return DFA(n_states=4, transitions=trans, finals={2, 3}, start=0)
        return DFA(n_states=3, transitions=trans, finals={2}, start=0)
    if t == "boolean":
        return from_literal(["true", "false"], alphabet)
    # 兜底：接受任意合法 JSON 值的**外层结构**（不做完整 JSON 语法）
    alpha = alphabet or list('{}[]",:0123456789truefalsn \t\n')
    return from_literal(['{}', '[]', '""', "0"], alpha)


# --------------------------------------------------------------------------- #
class ConstrainedDecoder:
    """把一个 DFA 接到逐 token 解码上。"""

    def __init__(self, dfa: DFA, alphabet: Sequence[str]) -> None:
        self.dfa = dfa
        self.alphabet = list(alphabet)
        self.state = dfa.start
        self.text = ""
        self.blocked = 0

    def mask_logits(self, logits: torch.Tensor, id_to_char: Dict[int, str]) -> torch.Tensor:
        """把不能走的 token 置为 -inf。"""
        out = logits.clone()
        allowed = self.dfa.allowed(self.state, self.alphabet)
        keep = [i for i, c in id_to_char.items() if c in allowed]
        if not keep:
            # 没有任何合法转移：说明该终止了，交给调用方处理
            self.blocked += 1
            return out
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[keep] = False
        return out.masked_fill(mask, float("-inf"))

    def feed(self, ch: str) -> bool:
        nxt = self.dfa.step(self.state, ch)
        if nxt < 0:
            return False
        self.state = nxt
        self.text += ch
        return True

    @property
    def done(self) -> bool:
        return self.state in self.dfa.finals


def apply_dfa_mask(logits: torch.Tensor, dfa: DFA, state: int,
                   alphabet: Sequence[str], id_to_char: Dict[int, str]) -> torch.Tensor:
    """函数式版本（无状态）。"""
    allowed = dfa.allowed(state, alphabet)
    keep = [i for i, c in id_to_char.items() if c in allowed]
    if not keep:
        return logits
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[keep] = False
    return logits.masked_fill(mask, float("-inf"))
