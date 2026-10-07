"""KV Cache 的 CPU 交换空间（Swap）。

vLLM 的抢占有两种策略：

* **recompute**（重计算）：把块直接释放，以后重新 prefill。
  简单，但要重算一遍 —— 长 prompt 时代价很高。
* **swap**：把被抢占序列的 KV 块搬到 **CPU 内存**，需要时再搬回来。
  显存压力大时它明显更划算（PCIe 单向 ~16 GB/s，
  搬 1GB KV 约 60ms，而重算一个 8k prompt 往往要几百 ms）。

代价：需要预留一块 CPU 内存，并且 swap in/out 会占用 PCIe 带宽，
所以真实的调度器会在"重算更便宜"时自动选 recompute。

本模块只做**块的搬运**，策略选择放在 ``scheduler`` 里（``preemption_mode``）。
"""

from __future__ import annotations

from typing import Dict, List

import torch

__all__ = ["CPUSwapSpace"]


class CPUSwapSpace:
    """GPU ↔ CPU 的 KV 块交换区。

    :param gpu_k / gpu_v: 形状 ``[num_blocks, block_size, Hkv, D]`` 的 GPU KV Cache
    :param num_blocks:    预留的 CPU 块数量
    """

    def __init__(self, gpu_k: torch.Tensor, gpu_v: torch.Tensor,
                 num_blocks: int, device: str = "cpu", block_dim: int = 1) -> None:
        """
        :param block_dim: 块所在的维度。分页 KV Cache 的布局是
            ``[n_layers, num_blocks, block_size, n_kv_heads, head_dim]``，所以默认 1。
        """
        if gpu_k.shape != gpu_v.shape:
            raise ValueError("K/V cache 形状必须一致")
        self.gpu_k = gpu_k
        self.gpu_v = gpu_v
        self.block_dim = int(block_dim)
        self.num_blocks = int(num_blocks)
        self.free_blocks = list(range(self.num_blocks))
        rest = list(gpu_k.shape)
        rest.pop(self.block_dim)
        shape = (self.num_blocks, *rest)
        self.k = torch.zeros(shape, dtype=gpu_k.dtype, device=device)
        self.v = torch.zeros(shape, dtype=gpu_v.dtype, device=device)
        self.stats = {"swap_out_blocks": 0, "swap_in_blocks": 0,
                      "swap_out_bytes": 0, "swap_in_bytes": 0}

    # ------------------------------------------------------------------ #
    def _gpu_slice(self, cache: torch.Tensor, blk: int) -> torch.Tensor:
        idx = [slice(None)] * cache.dim()
        idx[self.block_dim] = blk
        return cache[tuple(idx)]

    # ------------------------------------------------------------------ #
    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def _take(self, n: int) -> List[int]:
        if n > len(self.free_blocks):
            raise RuntimeError(f"CPU 交换空间不足：需要 {n}，剩余 {len(self.free_blocks)}")
        out = self.free_blocks[:n]
        del self.free_blocks[:n]
        return out

    def _give(self, slots: List[int]) -> None:
        self.free_blocks.extend(slots)

    # ------------------------------------------------------------------ #
    def swap_out(self, gpu_block_ids: List[int]) -> List[int]:
        """GPU → CPU。返回这批块占用的 CPU slot（调用方要自己记住）。"""
        if not gpu_block_ids:
            return []
        slots = self._take(len(gpu_block_ids))
        bytes_per_block = self.k[0].numel() * self.k.element_size()
        for slot, blk in zip(slots, gpu_block_ids):
            self.k[slot].copy_(self._gpu_slice(self.gpu_k, blk), non_blocking=True)
            self.v[slot].copy_(self._gpu_slice(self.gpu_v, blk), non_blocking=True)
        self.stats["swap_out_blocks"] += len(gpu_block_ids)
        self.stats["swap_out_bytes"] += 2 * len(gpu_block_ids) * bytes_per_block
        return slots

    def swap_in(self, slots: List[int], gpu_block_ids: List[int]) -> None:
        """CPU → GPU，把之前换出的块搬回指定 GPU 块。"""
        if not slots:
            return
        if len(slots) != len(gpu_block_ids):
            raise ValueError("slots 与 gpu_block_ids 数量必须一致")
        bytes_per_block = self.k[0].numel() * self.k.element_size()
        for slot, blk in zip(slots, gpu_block_ids):
            self._gpu_slice(self.gpu_k, blk).copy_(self.k[slot], non_blocking=True)
            self._gpu_slice(self.gpu_v, blk).copy_(self.v[slot], non_blocking=True)
        self.stats["swap_in_blocks"] += len(slots)
        self.stats["swap_in_bytes"] += 2 * len(slots) * bytes_per_block
        self._give(slots)

    def free(self, slots: List[int]) -> None:
        self._give(list(slots))

    def reset_stats(self) -> None:
        self.stats = {k: 0 for k in self.stats}
