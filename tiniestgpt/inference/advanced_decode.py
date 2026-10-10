"""进阶解码：树状投机解码 + Beam Search。

**树状投机（EAGLE-2/3、Medusa 的核心）**
链式投机一次只赌一条路径，接受率随长度指数衰减。
树状投机同时保留**多条候选分支**，用一棵树组织它们，
然后一次前向验证整棵树——因为共用前缀，验证成本只比一条链多一点点。

关键：验证时必须**逐节点检查因果一致性**——
某个节点能被接受，前提它的祖先都被接受。

**Beam Search**
对"有唯一正确答案"的任务（翻译、代码、结构化抽取）比采样更稳，
但会带来两个副作用：输出多样性差、且容易出现"退化重复"。
另外：对**递归状态模型**（GDN/Mamba/RWKV）beam 的代价更高——
每个 beam 都要复制或分叉一份状态，不能像 KV Cache 那样简单共享前缀。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["TreeSpecConfig", "build_draft_tree", "verify_tree", "TreeSpecStats",
           "beam_search", "BeamHypothesis"]


@dataclass
class TreeSpecConfig:
    n_draft: int = 8            # 树里的候选节点总数
    branching: List[int] = field(default_factory=lambda: [1, 2, 2, 2])  # 每层分支数
    top_k: int = 4              # 每个节点取 top-k 扩展


@dataclass
class TreeSpecStats:
    nodes: int = 0
    accepted: int = 0
    rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.nodes, 1)

    def report(self) -> str:
        return (f"tree-spec: 节点 {self.nodes}，接受 {self.accepted} "
                f"(α={self.acceptance_rate:.2%})，{self.rounds} 轮")


def build_draft_tree(draft_logits: Sequence[torch.Tensor], cfg: TreeSpecConfig
                     ) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """从 draft 模型的 logits 序列构建候选树。

    :return: ``(edges, node_tokens)``，``edges`` 为 ``(parent_idx, token)``
    """
    edges: List[Tuple[int, int]] = []
    tokens: List[int] = []
    level_sizes = [1]
    prev_level = [-1]                       # 根的父节点
    for depth, n_branch in enumerate(cfg.branching):
        if depth >= len(draft_logits) or len(tokens) >= cfg.n_draft:
            break
        lg = draft_logits[depth]
        probs = torch.softmax(lg.float(), dim=-1)
        topv, topi = torch.topk(probs, min(cfg.top_k, probs.numel()))
        cur: List[int] = []
        for p in prev_level:
            for j in range(min(n_branch, topi.numel())):
                if len(tokens) >= cfg.n_draft:
                    break
                edges.append((p, int(topi[j])))
                tokens.append(int(topi[j]))
                cur.append(len(tokens) - 1)
        if not cur:
            break
        prev_level = cur
        level_sizes.append(len(cur))
    return edges, torch.tensor(tokens, dtype=torch.long)


def verify_tree(target_logits: torch.Tensor, edges: List[Tuple[int, int]],
                stats: Optional[TreeSpecStats] = None) -> List[int]:
    """用 target 模型的一次前向验证整棵树，返回**最长可接受的链**。

    简化判定：节点 t 被接受当且仅当 target 的 argmax 等于该节点的 token
    （等价于 greedy 下的接受-拒绝）。采样场景需换成概率比较。
    """
    stats = stats or TreeSpecStats()
    stats.nodes = len(edges)
    stats.rounds += 1
    accepted_mask = [False] * len(edges)
    chosen = [int(torch.argmax(target_logits[i])) for i in range(min(len(edges),
                                                                     target_logits.shape[0]))]
    for i, (parent, tok) in enumerate(edges):
        if i >= len(chosen):
            break
        ancestor_ok = (parent < 0) or accepted_mask[parent]
        if ancestor_ok and chosen[i] == tok:
            accepted_mask[i] = True
            stats.accepted += 1
    # 取最长的一条可接受路径
    best: List[int] = []
    for i, ok in enumerate(accepted_mask):
        if not ok:
            continue
        chain, cur = [], i
        while cur >= 0 and accepted_mask[cur]:
            chain.append(edges[cur][1])
            cur = edges[cur][0]
        if len(chain) > len(best):
            best = list(reversed(chain))
    return best


# --------------------------------------------------------------------------- #
@dataclass
class BeamHypothesis:
    tokens: List[int]
    logprob: float
    finished: bool = False

    @property
    def score(self) -> float:
        """长度归一化后的得分（不归一化会偏爱短句）。"""
        return self.logprob / max(len(self.tokens), 1) ** 0.7


def beam_search(step_fn, prompt_ids: Sequence[int], num_beams: int = 4,
                max_new_tokens: int = 32, eos_id: int = 1,
                length_penalty: float = 0.7) -> List[BeamHypothesis]:
    """通用 beam search。``step_fn(tokens) -> logits[V]``（只需最后一个位置）。

    对递归状态模型要传 ``clone_state``，否则每个 beam 会共享同一份状态。
    """
    beams = [BeamHypothesis(tokens=list(prompt_ids), logprob=0.0)]
    results: List[BeamHypothesis] = []
    for _ in range(max_new_tokens):
        cands: List[BeamHypothesis] = []
        for b in beams:
            if b.finished:
                results.append(b)
                continue
            logits = step_fn(b.tokens)
            lp = torch.log_softmax(logits.float(), dim=-1)
            topv, topi = torch.topk(lp, num_beams)
            for v, t in zip(topv.tolist(), topi.tolist()):
                cands.append(BeamHypothesis(tokens=b.tokens + [int(t)],
                                            logprob=b.logprob + float(v),
                                            finished=(int(t) == eos_id)))
        if not cands:
            break
        cands.sort(key=lambda h: h.score, reverse=True)
        beams = [h for h in cands if not h.finished][:num_beams]
        results.extend([h for h in cands if h.finished][:num_beams])
        if not beams:
            break
    results.extend(beams)
    results.sort(key=lambda h: h.score, reverse=True)
    return results[:num_beams]
