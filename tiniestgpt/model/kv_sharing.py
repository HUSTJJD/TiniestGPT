"""KV Cache 的两条压缩路线：**K=V 共享**与**跨层 KV Sharing**。

**K=V 共享**（DeepSeek-V4 的 long-range MQA、Gemma 4 global）::

    标准:  Cache = [K, V]          两份
    K=V:   Cache = [C]，K ← C，V ← C   一份

既省 KV Cache 一半，也省压缩池的写入带宽；代价是 K 与 V 的表达被耦合，
所以采用者会用更宽的 head、低秩输出投影或局部分支来补偿表达能力。

**跨层 KV Sharing**（Gemma 4 E2B/E4B）::

    Producer layer: K,V = project(x)；写入 cache slot
    Consumer layer: Q = project(x)；**读 producer 的 slot**，自己不投影 K/V

E2B 的 35 层里只有 15 个 KV producer，20 层复用 → KV Cache slot 降到 42.9%。
注意不同层的 Q 仍然不同，所以 attention output 不是简单复制。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["build_kv_share_plan", "KVSharePlan", "KVSharingMixin"]


class KVSharePlan:
    """哪些层是 producer、哪些层复用谁的 KV。

    :param pattern: 逗号分隔的 ``p``(producer) / ``c``(consumer)，如 ``"p,c,c,p,c"``
    """

    def __init__(self, n_layers: int, pattern: str = "") -> None:
        pat = [c.strip().lower() for c in pattern.split(",") if c.strip()]
        self.owner: List[int] = []
        if not pat:
            self.owner = list(range(n_layers))          # 全部 producer
        else:
            last_p = 0
            for i in range(n_layers):
                ch = pat[i % len(pat)]
                if ch == "p":
                    self.owner.append(i)
                    last_p = i
                else:
                    self.owner.append(last_p)
        self.n_producers = len(set(self.owner))
        self.n_layers = n_layers

    def is_producer(self, layer_idx: int) -> bool:
        return self.owner[layer_idx] == layer_idx

    def owner_of(self, layer_idx: int) -> int:
        return self.owner[layer_idx]

    @property
    def slot_ratio(self) -> float:
        return self.n_producers / max(self.n_layers, 1)

    def report(self) -> str:
        return (f"KVSharing: {self.n_producers}/{self.n_layers} 个 producer "
                f"({self.slot_ratio:.1%} slots)  pattern={self.owner}")


def build_kv_share_plan(n_layers: int, pattern: str = "") -> KVSharePlan:
    return KVSharePlan(n_layers, pattern)


class KVSharingMixin:
    """给 Attention 加上"复用上一层 KV"的能力。

    约定：mixin 的宿主需要有 ``k_proj`` / ``v_proj`` / ``layer_idx``。
    consumer 层**不调用** k_proj/v_proj（它们仍存在，但只为参数量兼容；
    真正省的是 cache slot 与投影 FLOPs，所以要显式跳过 forward）。
    """

    plan: Optional[KVSharePlan] = None

    def _should_project_kv(self, layer_idx: int) -> bool:
        if self.plan is None:
            return True
        return self.plan.is_producer(layer_idx)

    def _kv_owner(self, layer_idx: int) -> int:
        return layer_idx if self.plan is None else self.plan.owner_of(layer_idx)

    def _project_kv(self, x: torch.Tensor, layer_idx: int):
        """返回 ``(k, v, owner_layer)``；consumer 层返回 ``(None, None, owner)``。"""
        if self._should_project_kv(layer_idx):
            return self.k_proj(x), self.v_proj(x), layer_idx
        return None, None, self._kv_owner(layer_idx)
