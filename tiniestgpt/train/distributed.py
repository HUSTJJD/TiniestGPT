"""分布式训练封装：DDP 与 FSDP。

选择依据：
  * **DDP**：每张卡放一份完整参数 + 优化器状态，反向时 all-reduce 梯度。
    简单、通信量固定（2·(N-1)/N · 参数量），但显存冗余大。
  * **FSDP**：把参数 / 梯度 / 优化器状态**分片**到所有卡，
    前向时按需 all-gather、用完立即释放。显存占用近乎除以 N，
    代价是额外的通信与实现复杂度。

关键优化：
  * ``gradient_as_bucket_view`` + 大 bucket → 通信与计算重叠；
  * ``static_graph=True``（图结构不变时）省掉每步的 bucket 重建；
  * FSDP 的 ``MixedPrecision`` 让参数保持 fp32、计算用 bf16。
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

__all__ = ["init_distributed", "destroy_distributed", "wrap_model", "reduce_metrics",
           "is_distributed", "barrier"]


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """返回 (rank, world_size, local_rank)。未启用时返回 (0, 1, 0)。"""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def wrap_model(model: nn.Module, strategy: str = "ddp", cfg=None) -> nn.Module:
    """按策略包装模型；单卡时原样返回。"""
    if not is_distributed():
        return model
    if strategy == "ddp":
        bucket_mb = getattr(cfg, "bucket_cap_mb", 25) if cfg else 25
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            bucket_cap_mb=bucket_mb,
            gradient_as_bucket_view=True,      # 省一次拷贝，且利于通信重叠
            static_graph=True,                 # 结构固定时可省掉每步重建 bucket
            broadcast_buffers=False,
        )
    if strategy == "fsdp":
        from torch.distributed.fsdp import (
            BackwardPrefetch, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from ..model.transformer import TransformerBlock

        mp = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        sharding = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "noop": ShardingStrategy.NO_SHARD,
        }.get(getattr(cfg, "fsdp_sharding", "full") if cfg else "full",
              ShardingStrategy.FULL_SHARD)
        return FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
            sharding_strategy=sharding,
            mixed_precision=mp,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,   # 预取下一层的 all-gather
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        )
    return model


def reduce_metrics(metrics: Dict[str, float], device: Optional[torch.device] = None) -> Dict[str, float]:
    """跨卡平均指标（用于日志）。"""
    if not is_distributed():
        return metrics
    dev = device or (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    out = {}
    for k, v in metrics.items():
        t = torch.tensor([float(v)], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = float(t.item()) / dist.get_world_size()
    return out
