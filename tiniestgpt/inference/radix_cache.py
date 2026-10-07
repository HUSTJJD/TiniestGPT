"""Radix Tree 前缀缓存：把"前缀复用"从哈希表升级成**可淘汰的前缀树**。

``scheduler.PrefixCache`` 是 ``hash(块内容) → block_id`` 的平铺字典，
能命中"从第一个 token 开始的完全相同前缀"，但有两个硬伤：

1. **没有淘汰策略** —— 表里只会越积越多，块永远不会被回收；
2. **表达不了"分叉共享"** —— 同一个系统提示词下挂 100 个不同问题，
   哈希方案要为每个请求各存一份完整前缀的映射，
   而树形结构里共享的那段前缀只存一次。

本实现是一棵以**块为粒度**的前缀树（内部节点恰好覆盖 ``block_size`` 个 token，
末尾节点可以不满一块），并带 LRU 淘汰：

* ``match_prefix(ids)`` → ``(可复用的 block_id 列表, 命中 token 数)``
* ``insert(ids, block_ids)`` → 把新算出来的块挂到树上
* ``evict(n)`` → 按"引用计数为 0 + 最久未访问"淘汰，返回被释放的 block_id

与 SGLang 的 RadixAttention 相比，这里没有做 token 级的节点分裂
（那需要更细的锁与内存管理），但**淘汰 + 分叉共享**这两个核心收益是完整的。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["RadixTreeNode", "RadixPrefixCache"]


@dataclass
class RadixTreeNode:
    """前缀树节点：一条边 = 一段 token，节点上挂着它覆盖的物理块。"""

    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixTreeNode"] = field(default_factory=dict)
    block_ids: List[int] = field(default_factory=list)
    parent: Optional["RadixTreeNode"] = None
    refcount: int = 0
    last_access: float = field(default_factory=time.time)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Node n_tok={len(self.tokens)} blocks={self.block_ids} rc={self.refcount}>"


class RadixPrefixCache:
    """块级 radix 树 + LRU 淘汰。"""

    def __init__(self, block_size: int = 16, enabled: bool = True) -> None:
        self.block_size = block_size
        self.enabled = enabled
        self.root = RadixTreeNode()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._nodes: List[RadixTreeNode] = []

    # ------------------------------------------------------------------ #
    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.hits + self.misses, 1)

    def stats(self) -> Dict[str, float]:
        return {"hits": float(self.hits), "misses": float(self.misses),
                "evictions": float(self.evictions), "hit_rate": self.hit_rate,
                "nodes": float(len(self._nodes))}

    # ------------------------------------------------------------------ #
    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """沿树尽可能深地匹配，返回 ``(block_id 列表, 命中 token 数)``。"""
        if not self.enabled or len(token_ids) < self.block_size:
            return [], 0

        bs = self.block_size
        node = self.root
        blocks: List[int] = []
        matched = 0
        for start in range(0, len(token_ids) - bs + 1, bs):
            chunk = token_ids[start:start + bs]
            child = node.children.get(chunk[0])
            if child is None or child.tokens != chunk or not child.block_ids:
                break
            node = child
            node.last_access = time.time()
            blocks.extend(node.block_ids)
            matched += len(child.tokens)
            if len(child.tokens) < bs:       # 末尾不满一块 → 后面没得匹配了
                break
            if len(child.tokens) > bs:       # 理论上不会发生
                break

        self.hits += 1 if matched else 0
        self.misses += 0 if matched else 1
        return blocks, matched

    # ------------------------------------------------------------------ #
    def insert(self, token_ids: List[int], block_ids: List[int]) -> None:
        """把 ``token_ids`` 对应的 ``block_ids`` 挂到树上（覆盖整块的部分）。"""
        if not self.enabled or not token_ids:
            return

        bs = self.block_size
        node = self.root
        cursor = 0
        for start in range(0, len(token_ids), bs):
            chunk = token_ids[start:start + bs]
            child = node.children.get(chunk[0])
            if child is None:
                child = RadixTreeNode(tokens=list(chunk), parent=node)
                node.children[chunk[0]] = child
                self._nodes.append(child)
            elif child.tokens != chunk:
                # 同一首 token 但内容不同 → 拆开重建（罕见路径）
                child.tokens = list(chunk)
                child.block_ids = []
                child.children = {}
                child.refcount = 0
            node = child
            node.last_access = time.time()

            if len(chunk) == bs and cursor < len(block_ids):
                if block_ids[cursor] not in node.block_ids:
                    node.block_ids.append(block_ids[cursor])
                cursor += 1
            if len(chunk) < bs:
                break

    # ------------------------------------------------------------------ #
    def evict(self, n_blocks: int) -> List[int]:
        """按 LRU 淘汰，返回被释放的 block_id（至少 ``n_blocks`` 个，能放多少放多少）。"""
        if n_blocks <= 0:
            return []
        candidates = [n for n in self._nodes if n.refcount <= 0 and n.block_ids]
        candidates.sort(key=lambda n: n.last_access)

        freed: List[int] = []
        for node in candidates:
            if len(freed) >= n_blocks:
                break
            freed.extend(node.block_ids)
            node.block_ids.clear()
            self.evictions += 1
            if node.is_leaf and node.parent is not None:
                node.parent.children.pop(node.tokens[0], None)
                if node in self._nodes:
                    self._nodes.remove(node)
        return freed

    def clear(self) -> None:
        self.root = RadixTreeNode()
        self._nodes.clear()
