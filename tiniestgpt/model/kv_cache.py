"""KV Cache：推理系统的核心数据结构。

这里把两种主流形态都实现了：

* **DenseKVCache**——每个序列一段连续显存 ``[B, max_seq, H_kv, D]``。
  实现简单，但**必须按最大长度预先分配**，且序列结束后显存不能给别人用
  → 显存碎片 + 浪费（vLLM 论文里指出实际利用率常常 < 50%）。

* **PagedKVCache**——借鉴操作系统虚拟内存的分页思想：
  把 KV 切成固定大小 ``block_size`` 的块，用 **block table** 做逻辑→物理映射。
  好处：按需分配（几乎零浪费）、天然支持 prefix caching / beam search 共享前缀、
  支持抢占与重计算。

**为什么 KV Cache 是推理的第一性问题**：
decode 阶段每生成一个 token 都要读一遍完整 KV，
``显存占用 = 2 · layers · n_kv_heads · head_dim · seq_len · batch · dtype``，
``带宽需求 ∝ 同样的量``。因此推理优化的主线就是：
减少 KV 的体积（GQA / MLA / 量化）与提高 KV 的访问效率（PagedAttention / 内核融合 / 批处理）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

__all__ = ["CacheView", "KVCache", "DenseKVCache", "PagedKVCache", "BlockAllocator"]


@dataclass
class CacheView:
    """一次 attention 计算所需的 KV 视图（duck typing，模型不关心具体实现）。"""
    kind: str                                  # "dense" | "paged"
    k: torch.Tensor                            # dense: [B, Tmax, Hkv, D]  paged: [NB, BS, Hkv, D]
    v: Optional[torch.Tensor] = None           # MLA 场景下为 None（K/V 共享一条 latent）
    block_table: Optional[torch.Tensor] = None # [B, max_blocks_per_seq]
    seq_lens: Optional[torch.Tensor] = None    # [B] 每个序列当前总长度（含刚写入的 token）
    kv_len: int = 0                            # dense 路径下已缓存的长度
    window: int = -1
    sinks: int = 0
    # KV 量化（kind="paged_quant"）时的缩放因子
    k_scale: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None
    # CUDA Graph 需要"长度也是静态的"：>0 时内核直接用它，避免 .item() 造成的 host 同步
    max_len: int = -1


class KVCache:
    """所有 KV Cache 的统一协议。"""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # 供线性注意力/RNN 类层存放循环状态（layer_idx -> tensor）
        self.states: Dict[int, torch.Tensor] = {}

    # 引擎在每次 forward 前调用，告诉缓存"这一批是什么"
    def set_batch(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, "_batch_" + k, v)

    def write(self, layer_idx: int, k: torch.Tensor, v: Optional[torch.Tensor]) -> CacheView:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #
class DenseKVCache(KVCache):
    """连续显存 KV Cache，适合 batch 内长度一致的自回归生成。"""

    def __init__(self, n_layers: int, batch_size: int, max_seq_len: int,
                 n_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float32,
                 device: torch.device | str = "cpu") -> None:
        super().__init__(n_layers, n_kv_heads, head_dim)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self._dtype = dtype
        shape = (n_layers, batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.write_pos = 0
        self.window = -1
        self.sinks = 0

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_batch(self, window: int = -1, sinks: int = 0, **kwargs) -> None:
        self.window = window
        self.sinks = sinks

    def write(self, layer_idx: int, k: torch.Tensor, v: Optional[torch.Tensor]) -> CacheView:
        # k: [B, T, Hkv, D]
        T = k.shape[1]
        start = self.write_pos
        end = start + T
        if end > self.max_seq_len:
            raise RuntimeError(f"KV Cache 溢出: {end} > {self.max_seq_len}")
        self.k_cache[layer_idx, :, start:end].copy_(k)
        if v is not None:
            self.v_cache[layer_idx, :, start:end].copy_(v)
        return CacheView(
            kind="dense",
            k=self.k_cache[layer_idx, :, :end],
            v=None if v is None else self.v_cache[layer_idx, :, :end],
            kv_len=end,
            window=self.window,
            sinks=self.sinks,
        )

    def advance(self, n: int) -> None:
        """写入 n 个 token 后推进写指针（由引擎在 forward 结束后调用）。"""
        self.write_pos += n

    def reset(self) -> None:
        self.write_pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.states.clear()

    @property
    def memory_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2


# --------------------------------------------------------------------------- #
# 块分配器（PagedAttention 的"内存管理单元"）
# --------------------------------------------------------------------------- #
class BlockAllocator:
    """空闲块栈 + 引用计数。

    * ``allocate(n)``：分配 n 个物理块
    * ``free(ids)``  ：归还
    * 引用计数 > 1 时（prefix caching / beam 共享）不真正释放
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))
        self.refcount = [0] * num_blocks

    def allocate(self, n: int = 1) -> List[int]:
        if len(self.free_blocks) < n:
            raise RuntimeError(f"KV 块不足：需要 {n}，剩余 {len(self.free_blocks)}")
        out = [self.free_blocks.pop() for _ in range(n)]
        for b in out:
            self.refcount[b] = 1
        return out

    def free(self, ids: List[int]) -> None:
        for b in ids:
            if b < 0:
                continue
            self.refcount[b] -= 1
            if self.refcount[b] <= 0:
                self.refcount[b] = 0
                self.free_blocks.append(b)

    def incr_ref(self, ids: List[int]) -> None:
        for b in ids:
            if b >= 0:
                self.refcount[b] += 1

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    @property
    def usage(self) -> float:
        return 1.0 - len(self.free_blocks) / max(self.num_blocks, 1)


# --------------------------------------------------------------------------- #
# Paged
# --------------------------------------------------------------------------- #
class PagedKVCache(KVCache):
    """分页 KV Cache：``k_cache[layer][block][slot][head][dim]``。"""

    def __init__(self, n_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float32,
                 device: torch.device | str = "cpu") -> None:
        super().__init__(n_layers, n_kv_heads, head_dim)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._dtype = dtype
        shape = (n_layers, num_blocks, block_size, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.allocator = BlockAllocator(num_blocks)
        # 批次元信息（由引擎 set_batch 注入）
        self.block_table: Optional[torch.Tensor] = None   # [B, max_blocks_per_seq] int32
        self.seq_lens: Optional[torch.Tensor] = None      # [B] int32
        self.slot_ids: Optional[torch.Tensor] = None      # [B, T_max] 或 [N]
        self.window = -1
        self.sinks = 0
        self.max_len = -1            # >0 时供 CUDA Graph 使用（静态长度）

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_batch(self, block_table=None, seq_lens=None, slot_ids=None,
                  window: int = -1, sinks: int = 0, max_len: int = -1) -> None:
        if block_table is not None:
            self.block_table = block_table
        if seq_lens is not None:
            self.seq_lens = seq_lens
        if slot_ids is not None:
            self.slot_ids = slot_ids
        self.window = window
        self.sinks = sinks
        self.max_len = max_len

    def write(self, layer_idx: int, k: torch.Tensor, v: Optional[torch.Tensor]) -> CacheView:
        """按 slot_ids 把新 token 的 K/V 散列写入物理块。

        :param k/v: [B, T, Hkv, D]（prefill，按行 padded）或 [N, Hkv, D]（扁平）
        :param slot_ids: 与 k/v 的 token 一一对应的物理槽位
        """
        slot_ids = self.slot_ids
        if slot_ids is None:
            raise RuntimeError("PagedKVCache 需要 slot_ids，请先调用 set_batch()")

        if slot_ids.dim() == 2:
            B, T, H, D = k.shape
            flat_slots = slot_ids.reshape(-1)                     # [B*T]
            mask = flat_slots >= 0
            flat_k = k.reshape(-1, H, D)[mask]                    # [N, H, D]
            slots = flat_slots[mask]
        else:
            flat_k = k.reshape(-1, k.shape[-2], k.shape[-1])
            slots = slot_ids
            mask = slots >= 0
            flat_k = flat_k[mask]
            slots = slots[mask]

        self.k_cache[layer_idx].view(-1, self.n_kv_heads, self.head_dim).index_copy_(
            0, slots.long(), flat_k.to(self._dtype))
        if v is not None:
            flat_v = v.reshape(-1, v.shape[-2], v.shape[-1])
            if mask is not None and flat_v.shape[0] != flat_k.shape[0]:
                flat_v = flat_v[mask]
            self.v_cache[layer_idx].view(-1, self.n_kv_heads, self.head_dim).index_copy_(
                0, slots.long(), flat_v.to(self._dtype))
            v_view: Optional[torch.Tensor] = self.v_cache[layer_idx]
        else:
            v_view = None

        return CacheView(
            kind="paged",
            k=self.k_cache[layer_idx],
            v=v_view,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
            window=self.window,
            sinks=self.sinks,
            max_len=self.max_len,
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.allocator = BlockAllocator(self.num_blocks)
        self.block_table = None
        self.seq_lens = None
        self.slot_ids = None
        self.states.clear()

    @property
    def memory_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2
