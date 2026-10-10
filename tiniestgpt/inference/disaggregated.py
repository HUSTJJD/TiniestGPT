"""Prefill / Decode 解耦（PD 解耦，2025–2026 的成熟范式）。

同一个 GPU 上混跑 prefill 与 decode 有两个根本冲突：

* prefill 是**计算密集**、大 batch、跑一次就走；
* decode 是**带宽密集**、小 batch、要一直占着显存；

两者混在一起时，一条长 prefill 会把所有 decode 请求的 TPOT 拉爆；
而为了保 TPOT 又得限制 prefill 的 batch，吞吐又上不去。

解耦的做法是**分开部署**：prefill worker 算完把 KV 传给 decode worker。
关键工程点有三个：

1. **KV 传输**：量很大（128K 上下文几十 GB 级别），必须走高速互联，
   且要按层流水线传输，边算边传；
2. **调度**：哪个 decode 实例接手（要考虑它已有的 KV 显存与排队长度）；
3. **不总是更优**：短 prompt、低 QPS 时，传输开销 > 分离收益。

本模块提供单机模拟：两个 worker + 一个传输总线，把
「分离 vs 混合」的 TTFT/TPOT 差异量化出来。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["PDConfig", "KVBus", "PrefillWorker", "DecodeWorker", "DisaggregatedServer",
           "compare_disaggregated"]


@dataclass
class PDConfig:
    n_prefill: int = 1
    n_decode: int = 1
    kv_transfer_gbps: float = 50.0       # 模拟的互联带宽（GB/s）
    pipeline_layers: bool = True         # 边算边传（按层流水线）
    max_prefill_tokens: int = 8192


class KVBus:
    """KV 传输总线（模拟）：记录传输字节与耗时。"""

    def __init__(self, gbps: float = 50.0) -> None:
        self.gbps = gbps
        self.bytes = 0
        self.transfers = 0

    def send(self, kv_bytes: int) -> float:
        self.bytes += kv_bytes
        self.transfers += 1
        return kv_bytes / (self.gbps * 1e9)     # 秒


class PrefillWorker:
    def __init__(self, model, bus: KVBus, cfg: PDConfig) -> None:
        self.model = model
        self.bus = bus
        self.cfg = cfg
        self.stats = {"requests": 0, "tokens": 0, "seconds": 0.0}

    @torch.no_grad()
    def run(self, ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        t0 = time.time()
        logits = self.model(ids)
        n_kv = self.model.kv_cache_bytes_per_token() * ids.shape[1]
        self.stats["requests"] += 1
        self.stats["tokens"] += ids.numel()
        self.stats["seconds"] += time.time() - t0
        return logits, n_kv


class DecodeWorker:
    def __init__(self, model, bus: KVBus) -> None:
        self.model = model
        self.bus = bus
        self.stats = {"requests": 0, "tokens": 0, "transfer_seconds": 0.0}

    def accept(self, kv_bytes: int) -> float:
        dt = self.bus.send(kv_bytes)
        self.stats["transfer_seconds"] += dt
        self.stats["requests"] += 1
        return dt


class DisaggregatedServer:
    """把 prefill 与 decode 拆到不同 worker 上（单机模拟）。"""

    def __init__(self, model, cfg: Optional[PDConfig] = None) -> None:
        self.cfg = cfg or PDConfig()
        self.bus = KVBus(self.cfg.kv_transfer_gbps)
        self.prefill = [PrefillWorker(model, self.bus, self.cfg)
                        for _ in range(self.cfg.n_prefill)]
        self.decode = [DecodeWorker(model, self.bus) for _ in range(self.cfg.n_decode)]
        self._next = 0

    def handle(self, ids: torch.Tensor) -> Dict[str, float]:
        w = self.prefill[self._next % len(self.prefill)]
        self._next += 1
        _, kv_bytes = w.run(ids)
        d = self.decode[self._next % len(self.decode)]
        transfer = d.accept(kv_bytes)
        return {"kv_bytes": float(kv_bytes), "transfer_s": transfer}

    def report(self) -> str:
        p = self.prefill[0].stats
        return (f"PD 解耦: prefill {p['requests']} 请求 / {p['tokens']} token / "
                f"{p['seconds']:.2f}s；总线 {self.bus.bytes / 1e6:.1f} MB "
                f"({self.bus.transfers} 次)")


def compare_disaggregated(model, prompts: List[torch.Tensor],
                          cfg: Optional[PDConfig] = None) -> Dict[str, float]:
    """对比「分离部署」与「混合部署」的 TTFT / TPOT（模拟计时）。"""
    cfg = cfg or PDConfig()
    srv = DisaggregatedServer(model, cfg)

    t0 = time.time()
    ttft = []
    for p in prompts:
        r = srv.handle(p)
        ttft.append(time.time() - t0 + r["transfer_s"])
        t0 = time.time()

    t0 = time.time()
    with torch.no_grad():
        for p in prompts:
            model(p)
    mixed = time.time() - t0

    return {
        "separated_total_s": sum(ttft),
        "mixed_total_s": mixed,
        "transfer_overhead_s": sum(r["transfer_s"] for r in [srv.handle(prompts[0])]),
        "speedup": (mixed / sum(ttft)) if sum(ttft) > 0 else 0.0,
    }
