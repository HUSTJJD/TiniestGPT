"""多状态 Cache Manager（2026 混合架构的必需品）。

vLLM 的 PagedAttention 只管理 **KV Page**。2026 的模型在同一层序列里
混用 full / sliding / GDN / Mamba / RWKV，于是 cache manager 必须同时管五种状态：

    Paged KV      : Full / MLA / DSA 的 K/V 页
    Ring KV       : 滑动窗口层（只保留固定窗口，用环形缓冲）
    Recurrent     : GDN / Mamba / RWKV 的固定大小状态
    Shared KV     : Gemma 4 的 producer/consumer 映射
    Compressed    : DeepSeek-V4 CSA/HCA 的压缩池

而且 **prefix cache 也不再统一**：

* Full Attention：可以共享 KV page；
* GDN / Mamba：要缓存"前缀跑完时的状态快照"；
* DSA：还要额外缓存 indexer key 或压缩池；
* Shared KV：必须保持 layer donor 映射一致。

本模块给出统一入口 + 分层下沉（GPU → CPU → 磁盘）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["CacheKind", "MultiStateCache", "HierarchicalCache", "StateTier"]


class CacheKind:
    PAGED = "paged"
    RING = "ring"
    RECURRENT = "recurrent"
    SHARED = "shared"
    COMPRESSED = "compressed"


class StateTier:
    GPU = 0
    CPU = 1
    DISK = 2


@dataclass
class MultiStateCache:
    """按层类型分配状态容器，并把"命中/未命中"统计出来。"""

    n_layers: int
    layer_kinds: List[str]                       # 长度 n_layers，取值见 CacheKind
    block_size: int = 16
    n_kv_heads: int = 2
    head_dim: int = 64
    window: int = 0
    recurrent_shape: Optional[Tuple[int, ...]] = None
    device: torch.device = torch.device("cpu")

    paged: Dict[int, torch.Tensor] = field(default_factory=dict)
    ring: Dict[int, torch.Tensor] = field(default_factory=dict)
    recurrent: Dict[int, torch.Tensor] = field(default_factory=dict)
    compressed: Dict[int, torch.Tensor] = field(default_factory=dict)
    shared_owner: Dict[int, int] = field(default_factory=dict)
    snapshots: Dict[str, torch.Tensor] = field(default_factory=dict)

    stats: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        assert len(self.layer_kinds) == self.n_layers
        for i, k in enumerate(self.layer_kinds):
            if k == CacheKind.RECURRENT and self.recurrent_shape:
                self.recurrent[i] = torch.zeros(self.recurrent_shape, device=self.device)
            elif k == CacheKind.RING and self.window > 0:
                self.ring[i] = torch.zeros(2, self.window, self.n_kv_heads, self.head_dim,
                                           device=self.device)
            self.shared_owner[i] = i
        self.stats = {"paged_hit": 0, "paged_miss": 0, "snapshot": 0, "evict": 0}

    def kind_of(self, layer: int) -> str:
        return self.layer_kinds[layer]

    def bind_shared(self, consumer: int, producer: int) -> None:
        """Gemma 4 式跨层 KV 共享：consumer 层不分配自己的 slot。"""
        self.shared_owner[consumer] = producer
        self.ring.pop(consumer, None)
        self.paged.pop(consumer, None)

    def slot_count(self) -> int:
        """实际占用的 KV slot 数（跨层共享后会变少）。"""
        return len(set(self.shared_owner.values()))

    # ------------------------------------------------------------------ #
    def snapshot(self, prefix_key: str, states: Dict[int, torch.Tensor]) -> None:
        """缓存某个前缀跑完时的**递归状态**（线性层的前缀复用靠它）。"""
        for i, s in states.items():
            self.snapshots[f"{prefix_key}::{i}"] = s.detach().clone()
        self.stats["snapshot"] += 1

    def restore(self, prefix_key: str) -> Dict[int, torch.Tensor]:
        out = {}
        for k, v in self.snapshots.items():
            if k.startswith(f"{prefix_key}::"):
                out[int(k.split("::")[1])] = v.clone()
                self.stats["paged_hit"] += 1
        return out

    def report(self) -> str:
        kinds: Dict[str, int] = {}
        for k in self.layer_kinds:
            kinds[k] = kinds.get(k, 0) + 1
        return (f"MultiStateCache: {kinds}，KV slot {self.slot_count()}/{self.n_layers}，"
                f"快照 {len(self.snapshots)} 份，命中 {self.stats['paged_hit']}")


class HierarchicalCache:
    """分层 KV 缓存：GPU → CPU → 磁盘（LMCache / HiCache 思路）。

    长上下文场景下 KV 才是显存大户，把**近期不用**的 KV 下沉到 CPU/SSD，
    比单纯扩容 GPU 便宜一个数量级。代价是换回时要付 PCIe 带宽。
    """

    def __init__(self, device: torch.device = torch.device("cpu"),
                 cpu_bytes_budget: int = 1 << 30) -> None:
        self.device = device
        self.gpu: Dict[str, torch.Tensor] = {}
        self.cpu: Dict[str, torch.Tensor] = {}
        self.disk: Dict[str, str] = {}
        self.budget = cpu_bytes_budget
        self.cpu_used = 0
        self.stats = {"offload": 0, "fetch": 0, "miss": 0}
        self._lu: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def put(self, key: str, tensor: torch.Tensor) -> None:
        self.gpu[key] = tensor
        self._lu[key] = time.time()

    def offload(self, key: str) -> bool:
        if key not in self.gpu:
            return False
        t = self.gpu.pop(key).detach().to("cpu", copy=True)
        nbytes = t.numel() * t.element_size()
        while self.cpu_used + nbytes > self.budget and self.cpu:
            victim = min(self.cpu, key=lambda k: self._lu.get(k, 0.0))
            self.to_disk(victim)
        self.cpu[key] = t
        self.cpu_used += nbytes
        self.stats["offload"] += 1
        return True

    def to_disk(self, key: str) -> None:
        """教学实现：不做真实落盘，只记录"已下沉"并释放 CPU 内存。"""
        if key in self.cpu:
            self.cpu_used -= self.cpu[key].numel() * self.cpu[key].element_size()
            del self.cpu[key]
            self.disk[key] = f"<disk:{key}>"

    def fetch(self, key: str) -> Optional[torch.Tensor]:
        if key in self.gpu:
            return self.gpu[key]
        if key in self.cpu:
            t = self.cpu.pop(key).to(self.device)
            self.cpu_used -= t.numel() * t.element_size()
            self.gpu[key] = t
            self._lu[key] = time.time()
            self.stats["fetch"] += 1
            return t
        self.stats["miss"] += 1
        return None

    def report(self) -> str:
        return (f"HierarchicalCache: GPU {len(self.gpu)} / CPU {len(self.cpu)} "
                f"({self.cpu_used / 1e6:.0f} MB) / DISK {len(self.disk)}，"
                f"换入 {self.stats['fetch']}，换出 {self.stats['offload']}")
