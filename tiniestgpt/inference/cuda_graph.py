"""CUDA Graph：把 decode 一步的 **kernel launch 开销** 降到接近零。

decode 时每个 token 要跑几百个小 kernel（几十个 matmul + norm + rope + attn），
其中相当一部分时间花在 **launch 与 CPU 侧调度** 上，而不是 GPU 计算本身。
CUDA Graph 把"一步"录制成一个有向图，之后 **一次 replay 就提交全部 kernel**，
CPU 开销从 O(#kernels) 降到 O(1)，小 batch 下常见 1.2~2× 提速。

三个必须满足的条件（也是它难用的原因）：
  1. **形状固定**：输入张量的 shape 不能变 → 需要按 (batch_size, 长度档位) 分桶捕获；
  2. **指针固定**：张量对象本身不能重建 → 必须预先分配**静态缓冲区**，
     每步只把新数据 ``copy_`` 进去（包括 block_table / seq_lens / slot_ids）；
  3. **不能有 host 同步**：捕获期间不能出现 ``.item()`` / ``print(tensor)`` 之类，
     因此 KV 长度必须提前算好并以整数形式传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..model.kv_cache import PagedKVCache

__all__ = ["CUDAGraphManager", "StaticBatchCache"]


class StaticBatchCache(PagedKVCache):
    """与主缓存**共享物理显存**、但元信息是静态张量的缓存视图。

    这样 CUDA Graph 里访问到的 k_cache 指针与主引擎完全一致，
    只是 block_table / seq_lens / slot_ids 换成了固定地址的缓冲区。
    """

    def __init__(self, src: PagedKVCache) -> None:
        super().__init__(src.n_layers, src.num_blocks, src.block_size,
                         src.n_kv_heads, src.head_dim, dtype=src._dtype,
                         device=src.k_cache.device)
        self.k_cache = src.k_cache
        self.v_cache = src.v_cache
        self.allocator = src.allocator


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_table: torch.Tensor
    slot_ids: torch.Tensor
    seq_lens: torch.Tensor
    output: torch.Tensor
    max_len: int


class CUDAGraphManager:
    """按 (batch_size, 长度档位) 捕获多个图，forward 时自动选桶。"""

    def __init__(self, model, cache: PagedKVCache, batch_sizes: List[int],
                 len_buckets: Optional[List[int]] = None, max_blocks: int = 128) -> None:
        self.model = model
        self.cache = cache
        self.max_blocks = max_blocks
        self.len_buckets = sorted(len_buckets or [64, 256, 1024, 4096])
        self.entries: Dict[Tuple[int, int], _GraphEntry] = {}
        device = cache.k_cache.device
        if device.type != "cuda":
            raise RuntimeError("CUDA Graph 需要 CUDA 设备")
        for bs in batch_sizes:
            for L in self.len_buckets:
                try:
                    self.entries[(bs, L)] = self._capture(bs, L)
                except Exception as exc:      # 形状不支持 / 捕获失败 → 只跳过该桶
                    print(f"[cuda-graph] capture skip (bs={bs}, L={L}): {exc}")
        if not self.entries:
            raise RuntimeError("没有任何 CUDA Graph 捕获成功")

    # ------------------------------------------------------------------ #
    def _capture(self, bs: int, max_len: int) -> _GraphEntry:
        dev = self.cache.k_cache.device
        input_ids = torch.zeros((bs, 1), dtype=torch.long, device=dev)
        positions = torch.zeros((bs, 1), dtype=torch.long, device=dev)
        block_table = torch.full((bs, self.max_blocks), -1, dtype=torch.long, device=dev)
        slot_ids = torch.zeros((bs, 1), dtype=torch.long, device=dev)
        seq_lens = torch.full((bs,), max_len, dtype=torch.long, device=dev)

        scache = StaticBatchCache(self.cache)
        scache.set_batch(block_table=block_table, seq_lens=seq_lens, slot_ids=slot_ids,
                         max_len=max_len)

        # warmup（必须在 side stream 上，且要跑够次数让 cublas 等完成惰性初始化）
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.model(input_ids, positions=positions, cache=scache)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.model(input_ids, positions=positions, cache=scache)

        return _GraphEntry(graph, input_ids, positions, block_table, slot_ids, seq_lens,
                           out, max_len)

    # ------------------------------------------------------------------ #
    @property
    def graphs(self) -> Dict[Tuple[int, int], _GraphEntry]:
        return self.entries

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                block_table: torch.Tensor, slot_ids: torch.Tensor,
                seq_lens: torch.Tensor) -> Optional[torch.Tensor]:
        bs = input_ids.shape[0]
        need = int(seq_lens.max().item())
        bucket = next((L for L in self.len_buckets if L >= need), None)
        if bucket is None:
            return None
        e = self.entries.get((bs, bucket))
        if e is None:
            return None

        # 把所有动态数据 copy 进静态缓冲区（**不能重新赋值张量对象**）
        e.input_ids.copy_(input_ids)
        e.positions.copy_(positions)
        e.slot_ids.copy_(slot_ids)
        e.seq_lens.copy_(seq_lens)
        n = min(block_table.shape[1], self.max_blocks)
        e.block_table.zero_().add_(-1)
        e.block_table[:, :n].copy_(block_table[:, :n])
        e.graph.replay()
        return e.output
