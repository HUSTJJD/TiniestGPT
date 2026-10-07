"""KV Cache 量化：直接砍掉 decode 阶段的**显存与带宽**。

decode 每一步都要把整个 KV Cache 从 HBM 读一遍，
所以「KV 多大」几乎等价于「decode 多快」（在 batch 较大时尤其明显）。
把 KV 压到 int8 / fp8 通常能带来 **1.5~2×** 的 decode 提速，
而且精度损失远小于权重量化（因为 KV 的分布更平滑）。

两种粒度：
  * **per-token**：每个 token 一个 scale，最鲁棒（抵抗离群 token）；
  * **per-head**：每个 (token, head) 一组共享 scale，scale 表更小。

注意：实现上必须 **gather 之后再反量化**（见 ``paged_attention_quantized_ref``），
否则就退化成"先全量反量化"，白白浪费了省下来的带宽。
"""

from __future__ import annotations

from typing import Optional

import torch

from ...model.kv_cache import CacheView, PagedKVCache

__all__ = ["quantize_kv", "dequantize_kv", "QuantizedPagedKVCache"]


def quantize_kv(x: torch.Tensor, granularity: str = "per_token", bits: int = 8):
    """把 [N, H, D] 的 K/V 量化为 int8。

    scale 统一返回 ``[N, H, 1]`` 形状，便于与 gather 出来的 K/V 直接相乘：
      * ``per_head``：每个 (token, head) 一个 scale；
      * ``per_token``：每个 token 一个 scale（再广播到所有 head）。

    :return: (q int8 [N,H,D], scale fp16 [N,H,1])
    """
    qmax = 2 ** (bits - 1) - 1
    assert x.dim() == 3, "期望 [N, H, D]"
    if granularity == "per_head":
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax    # [N,H,1]
    else:
        # 对一个 token 的所有 (head, dim) 取一个 scale，再广播到每个 head
        scale = x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12) / qmax  # [N,1,1]
        scale = scale.expand(-1, x.shape[1], 1)
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_kv(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale


class QuantizedPagedKVCache(PagedKVCache):
    """分页 + int8 的 KV Cache。接口与 PagedKVCache 完全一致。"""

    def __init__(self, n_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float16,
                 device: torch.device | str = "cpu", granularity: str = "per_token",
                 bits: int = 8) -> None:
        super().__init__(n_layers, num_blocks, block_size, n_kv_heads, head_dim,
                         dtype=dtype, device=device)
        shape = (n_layers, num_blocks, block_size, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        self.granularity = granularity
        self.bits = bits
        # scale 形状固定为 [layer, block, slot, head, 1]，与 gather 出的 K/V 可直接相乘
        sc_shape = (n_layers, num_blocks, block_size, n_kv_heads, 1)
        self.k_scale = torch.zeros(sc_shape, dtype=torch.float16, device=device)
        self.v_scale = torch.zeros(sc_shape, dtype=torch.float16, device=device)

    # ------------------------------------------------------------------ #
    def write(self, layer_idx: int, k: torch.Tensor, v: Optional[torch.Tensor]) -> CacheView:
        slot_ids = self.slot_ids
        if slot_ids is None:
            raise RuntimeError("QuantizedPagedKVCache 需要 slot_ids，请先调用 set_batch()")

        if slot_ids.dim() == 2:
            B, T, H, D = k.shape
            flat_slots = slot_ids.reshape(-1)
            mask = flat_slots >= 0
            flat_k = k.reshape(-1, H, D)[mask]
            slots = flat_slots[mask]
            flat_v = v.reshape(-1, H, D)[mask] if v is not None else None
        else:
            flat_k = k.reshape(-1, k.shape[-2], k.shape[-1])
            slots = slot_ids
            mask = slots >= 0
            flat_k = flat_k[mask]
            slots = slots[mask]
            flat_v = v.reshape(-1, v.shape[-2], v.shape[-1])[mask] if v is not None else None

        slots = slots.long()
        kq, ks = quantize_kv(flat_k.float(), self.granularity, self.bits)
        self.k_cache[layer_idx].view(-1, self.n_kv_heads, self.head_dim).index_copy_(0, slots, kq)
        self.k_scale[layer_idx].view(-1, self.n_kv_heads, 1).index_copy_(0, slots, ks)

        if flat_v is not None:
            vq, vs = quantize_kv(flat_v.float(), self.granularity, self.bits)
            self.v_cache[layer_idx].view(-1, self.n_kv_heads, self.head_dim).index_copy_(0, slots, vq)
            self.v_scale[layer_idx].view(-1, self.n_kv_heads, 1).index_copy_(0, slots, vs)

        return CacheView(
            kind="paged_quant",
            k=self.k_cache[layer_idx], v=self.v_cache[layer_idx],
            k_scale=self.k_scale[layer_idx], v_scale=self.v_scale[layer_idx],
            block_table=self.block_table, seq_lens=self.seq_lens,
            window=self.window, sinks=self.sinks,
        )

    @property
    def memory_bytes(self) -> int:
        """int8 存储 + fp16 scale 表。"""
        base = self.k_cache.numel() + self.v_cache.numel()
        scales = (self.k_scale.numel() + self.v_scale.numel()) * 2
        return base + scales

    def compression_ratio(self, fp_dtype: torch.dtype = torch.float16) -> float:
        fp_bytes = self.k_cache.numel() * 2 * torch.empty(0, dtype=fp_dtype).element_size()
        return fp_bytes / max(self.memory_bytes, 1)
