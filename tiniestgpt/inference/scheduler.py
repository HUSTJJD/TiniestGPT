"""连续批处理调度器 + 分页块管理 + Prefix Caching。

为什么需要调度器：
不同请求的 prompt 长度不同、生成长度更不同。若像训练那样做**静态 batch**，
短请求必须等最长请求结束 → GPU 空转（这就是"尾延迟"的来源）。
**Continuous Batching** 的做法是：每一步都重新组 batch——
完成的请求立刻离开、等待的请求立刻补位，GPU 永远有活干。

配套的两个机制：
  * **Chunked Prefill**：把超长 prompt 切成若干块，与 decode 混合执行，
    避免一次 prefill 独占几百毫秒导致其它请求卡顿（并限制峰值显存）。
  * **Prefix Caching**：系统提示词 / 多轮对话前缀往往完全相同，
    按块哈希复用已算好的 KV 块，命中时**连计算都省了**。
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import torch

from ..model.kv_cache import BlockAllocator
from .radix_cache import RadixPrefixCache
from .swap import CPUSwapSpace

__all__ = ["SequenceStatus", "Sequence", "SchedulerOutput", "Scheduler", "BlockHasher",
           "PrefixCache", "RadixPrefixCache"]


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: List[int]
    sampling_params: "object"                      # 避免循环导入
    block_size: int = 16
    arrival_time: float = field(default_factory=time.time)

    output_ids: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    computed: int = 0                              # 已完成 prefill 的 token 数
    status: SequenceStatus = SequenceStatus.WAITING
    cum_logprob: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_ids)

    def get_len(self) -> int:
        """KV 中应有的 token 总数（prompt + 已生成）。"""
        return self.prompt_len + self.output_len

    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(int(token_id))

    def is_finished(self) -> bool:
        sp = self.sampling_params
        if getattr(self, "_force_finish", False):     # 约束解码：JSON 已闭合
            return True
        if self.output_len >= getattr(sp, "max_tokens", 64):
            return True
        if self.output_ids and self.output_ids[-1] in getattr(sp, "stop_token_ids", []):
            return not getattr(sp, "ignore_eos", False)
        return False

    def all_token_ids(self) -> List[int]:
        return list(self.prompt_ids) + list(self.output_ids)


@dataclass
class SchedulerOutput:
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    prefill_tokens: int = 0
    ignored_seqs: List[Sequence] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)


# --------------------------------------------------------------------------- #
# Prefix Caching
# --------------------------------------------------------------------------- #
class BlockHasher:
    """逐块哈希：``h_i = H(h_{i-1}, tokens_i)``。

    只哈希**完整的块**（最后一个不满的块不缓存），保证语义安全。
    """

    @staticmethod
    def hash_block(prev_hash: int, token_ids: List[int]) -> int:
        # 注意：不能用 bytes(token_ids) —— token id 常常 > 255（词表几万很正常），
        # 会直接抛 "bytes must be in range(0, 256)"。这里按 4 字节定长编码。
        h = hashlib.blake2b(digest_size=16)
        h.update(prev_hash.to_bytes(16, "little"))
        h.update(b"".join(int(t & 0xFFFFFFFF).to_bytes(4, "little") for t in token_ids))
        return int.from_bytes(h.digest(), "little")


class PrefixCache:
    """hash → block_id 的 LRU（简化：不淘汰，靠 refcount 与重置管理生命周期）。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.table: Dict[int, int] = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, h: int) -> Optional[int]:
        if not self.enabled:
            return None
        b = self.table.get(h)
        if b is None:
            self.misses += 1
        else:
            self.hits += 1
        return b

    def insert(self, h: int, block_id: int) -> None:
        if self.enabled:
            self.table[h] = block_id

    def clear(self) -> None:
        self.table.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


# --------------------------------------------------------------------------- #
# 调度器
# --------------------------------------------------------------------------- #
class Scheduler:
    def __init__(self, block_size: int = 16, max_num_seqs: int = 32,
                 max_num_batched_tokens: int = 4096, enable_chunked_prefill: bool = True,
                 enable_prefix_caching: bool = True, max_model_len: int = 2048,
                 prefix_cache_impl: str = "hash", preemption_mode: str = "recompute") -> None:
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_model_len = max_model_len
        self.prefix_cache = PrefixCache(enable_prefix_caching)
        # "hash"  = 平铺哈希表（默认，行为与旧版一致）
        # "radix" = 前缀树 + LRU 淘汰（见 inference/radix_cache.py）
        self.prefix_cache_impl = prefix_cache_impl
        self.radix_cache = RadixPrefixCache(block_size, enable_prefix_caching)
        # "recompute" = 抢占后重算；"swap" = 换出到 CPU 交换空间（需 set_swap_space）
        self.preemption_mode = preemption_mode
        self.swap_space: Optional[CPUSwapSpace] = None

        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []
        self.allocator: Optional[BlockAllocator] = None
        self.stats = {"preempted": 0, "swapped_out": 0, "evicted_blocks": 0,
                      "prefill_tokens": 0, "decode_tokens": 0, "steps": 0}

    def set_allocator(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator

    def set_swap_space(self, swap_space: Optional[CPUSwapSpace]) -> None:
        """挂上 CPU 交换空间后，抢占策略才会真正走 swap（否则自动回退 recompute）。"""
        self.swap_space = swap_space

    # ------------------------------------------------------------------ #
    def add_seq(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ #
    def _allocate_prefix(self, seq: Sequence) -> int:
        """利用 prefix caching 复用块，返回"已缓存、无需计算"的 token 数。"""
        assert self.allocator is not None
        if self.prefix_cache_impl == "radix":
            return self._allocate_prefix_radix(seq)
        cached_tokens = 0
        prev_hash = 0
        n_full = len(seq.prompt_ids) // self.block_size
        for i in range(n_full):
            chunk = seq.prompt_ids[i * self.block_size:(i + 1) * self.block_size]
            h = BlockHasher.hash_block(prev_hash, chunk)
            bid = self.prefix_cache.lookup(h)
            if bid is None or bid not in self.allocator.refcount or self.allocator.refcount[bid] <= 0:
                break
            self.allocator.incr_ref([bid])
            seq.block_table.append(bid)
            prev_hash = h
            cached_tokens += self.block_size
        seq._last_hash = prev_hash          # type: ignore[attr-defined]
        return cached_tokens

    def _allocate_prefix_radix(self, seq: Sequence) -> int:
        """Radix 树版前缀复用：命中即整段共享，未命中返回 0。"""
        assert self.allocator is not None
        blocks, matched = self.radix_cache.match_prefix(seq.prompt_ids)
        if not blocks:
            return 0
        for b in blocks:
            self.allocator.incr_ref([b])
        seq.block_table.extend(blocks)
        return matched

    def _ensure_blocks(self, seq: Sequence) -> bool:
        """为尚未缓存的部分分配物理块（失败返回 False → 抢占或等待）。"""
        assert self.allocator is not None
        # 被 swap 出去的序列：先把 KV 搬回来，再谈补块
        if getattr(seq, "_swap_slots", None):
            if not self._restore_swapped(seq):
                return False
        need = seq.n_blocks() - len(seq.block_table)
        if need <= 0:
            return True
        if self.allocator.num_free < need:
            # 先尝试淘汰 radix 缓存里没人用的块，再决定要不要抢占
            if self.prefix_cache_impl == "radix":
                freed = self.radix_cache.evict(need - self.allocator.num_free)
                if freed:
                    self.allocator.free(freed)
                    self.stats["evicted_blocks"] += len(freed)
            if self.allocator.num_free < need:
                return False
        new_blocks = self.allocator.allocate(need)
        seq.block_table.extend(new_blocks)
        # 记录新块的哈希（只记录完整块）
        prev_hash = getattr(seq, "_last_hash", 0)
        start = len(seq.block_table) - need
        for j, b in enumerate(new_blocks):
            bi = start + j
            lo, hi = bi * self.block_size, (bi + 1) * self.block_size
            if hi <= len(seq.prompt_ids):
                chunk = seq.prompt_ids[lo:hi]
                prev_hash = BlockHasher.hash_block(prev_hash, chunk)
                self.prefix_cache.insert(prev_hash, b)
                seq._last_hash = prev_hash   # type: ignore[attr-defined]
        if self.prefix_cache_impl == "radix":
            self.radix_cache.insert(seq.prompt_ids, list(seq.block_table))
        return True

    def free_seq(self, seq: Sequence) -> None:
        if self.allocator is not None and seq.block_table:
            self.allocator.free(list(seq.block_table))
        seq.block_table = []
        seq.computed = 0
        if self.swap_space is not None:
            slots = getattr(seq, "_swap_slots", None)
            if slots:
                self.swap_space.free(slots)
                seq._swap_slots = []        # type: ignore[attr-defined]

    def _preempt(self, seq: Sequence) -> None:
        """显存不足时把序列退回等待队列。

        * ``recompute``：直接释放块，下次从头重算（简单可靠，长 prompt 代价高）
        * ``swap``：把 KV 搬到 CPU 交换区，下次只搬回来（省掉重算）
        """
        if (self.preemption_mode == "swap" and self.swap_space is not None
                and seq.block_table and self.swap_space.num_free >= len(seq.block_table)):
            slots = self.swap_space.swap_out(list(seq.block_table))
            seq._swap_slots = slots         # type: ignore[attr-defined]
            if self.allocator is not None:
                self.allocator.free(list(seq.block_table))
            # 关键：块虽然归还了，但 **computed 不清零** —— KV 还在 CPU 上，
            # 重新调度时只需 swap in，不需要再算一遍 prefill。
            seq.block_table = []
            self.stats["swapped_out"] += len(slots)
        else:
            self.free_seq(seq)
        seq.status = SequenceStatus.PREEMPTED
        self.stats["preempted"] += 1
        self.waiting.appendleft(seq)

    def _restore_swapped(self, seq: Sequence) -> bool:
        """被换出的序列重新上调度时，把 KV 从 CPU 搬回 GPU 块。"""
        slots = getattr(seq, "_swap_slots", None)
        if not slots or self.swap_space is None or self.allocator is None:
            return False
        need = len(slots) + max(seq.n_blocks() - len(slots), 0)
        if self.allocator.num_free < need:
            return False
        new_blocks = self.allocator.allocate(need)
        self.swap_space.swap_in(slots, new_blocks[:len(slots)])
        seq.block_table.extend(new_blocks)
        seq._swap_slots = []                # type: ignore[attr-defined]
        return True

    # ------------------------------------------------------------------ #
    def schedule(self) -> SchedulerOutput:
        assert self.allocator is not None
        out = SchedulerOutput()
        self.stats["steps"] += 1

        # ---------- 1) decode 优先（保证正在生成的请求低延迟） ----------
        budget = self.max_num_batched_tokens
        still_running: List[Sequence] = []
        for seq in self.running:
            if len(out) >= self.max_num_seqs or budget <= 0:
                still_running.append(seq)
                continue
            if not self._ensure_blocks(seq):
                self._preempt(seq)
                continue
            out.decode_seqs.append(seq)
            budget -= 1
            still_running.append(seq)
        self.running = still_running

        # ---------- 2) 补充 prefill ----------
        admitted: List[Sequence] = []
        while self.waiting and len(out) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            if seq.status == SequenceStatus.PREEMPTED:
                seq.status = SequenceStatus.WAITING

            if seq.computed == 0:
                seq.computed = self._allocate_prefix(seq)
            if not self._ensure_blocks(seq):
                break                      # 块不够：本次不再接纳新请求

            total = seq.prompt_len
            remain = total - seq.computed
            if remain > budget:
                if not self.enable_chunked_prefill:
                    break
                remain = budget            # 分块预填充：只做一小块

            if not self.enable_chunked_prefill and remain < total - seq.computed:
                break

            out.prefill_seqs.append(seq)
            out.prefill_tokens += remain
            setattr(seq, "_chunk", remain)  # 本次要计算的 token 数
            budget -= remain
            admitted.append(seq)
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        for s in out.prefill_seqs:
            s.status = SequenceStatus.RUNNING
        return out

    # ------------------------------------------------------------------ #
    def finish_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.free_seq(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.finished.append(seq)

    def abort_all(self) -> None:
        for s in list(self.running) + list(self.waiting):
            self.free_seq(s)
        self.running.clear()
        self.waiting.clear()
