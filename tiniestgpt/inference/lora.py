"""LoRA 多租户热插拔：一份底座权重，同时服务 N 个适配器。

2026 的服务化场景里，"一个模型一个部署"是最贵的做法。
LoRA 让同一份底座权重同时挂多个 rank 很小的适配器（几 MB 级），
于是可以在线切换租户而**不重新加载底座**。

三个工程要点：

1. **merged vs unmerged**
   合并进底座（merge）推理最快，但切换租户要重新合并；
   不合并（BA 分开算）多一次小矩阵乘，但可以在 batch 内混用不同适配器。
2. **batch 内混用**
   不同请求可能用不同适配器，所以 LoRA 的 GEMM 必须**按请求分组**，
   或者退化为逐请求的小 GEMM（rank 很小时这个开销可接受）。
3. **显存上限与淘汰**
   适配器虽小，挂几百个也会占显存；要有 LRU 淘汰。

本模块给出：LoRA 层封装、merge/unmerge、**batch 内按适配器分组**的前向，
以及带容量上限的 LRU 适配器池。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRAConfig", "LoRALinear", "inject_lora", "LoRAPool",
           "grouped_lora_forward"]


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: Tuple[str, ...] = ("q_proj", "v_proj")

    @property
    def scale(self) -> float:
        return self.alpha / max(self.r, 1)


class LoRALinear(nn.Module):
    """``y = W x + (alpha/r) · B(A x)``，可随时 merge/unmerge。"""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.r, self.scale = cfg.r, cfg.scale
        self.lora_A = nn.Parameter(torch.zeros(cfg.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)     # B 置 0 → 初始等价原层
        self.merged = False
        self._merged_delta: Optional[torch.Tensor] = None

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self._merged_delta = self.delta()
        self.base.weight.data.add_(self._merged_delta)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.data.sub_(self._merged_delta)
        self._merged_delta = None
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged or self.r == 0:
            return out
        # BA x：先降到 r 维再升回去，rank 很小时开销可接受
        return out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.cfg.alpha}, merged={self.merged}"


def inject_lora(model: nn.Module, cfg: LoRAConfig,
                name_filter: Optional[callable] = None) -> Dict[str, LoRALinear]:
    """把模型里的 Linear 换成 LoRALinear（**共享底座权重对象**）。"""
    replaced: Dict[str, LoRALinear] = {}
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
            continue
        if not any(t in name for t in cfg.targets):
            continue
        if name_filter and not name_filter(name):
            continue
        parent_path, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_path) if parent_path else model
        new = LoRALinear(mod, cfg)
        setattr(parent, child, new)
        replaced[name] = new
    return replaced


def grouped_lora_forward(x: torch.Tensor, base_w: torch.Tensor,
                         adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
                         group_ids: torch.Tensor) -> torch.Tensor:
    """batch 内按适配器分组的前向：同组的请求合成一次 GEMM。

    ``group_ids`` 是 ``[N]`` 的适配器下标；-1 表示只用底座。
    """
    out = F.linear(x, base_w)
    for gid in torch.unique(group_ids):
        if gid < 0:
            continue
        sel = (group_ids == gid).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        name = str(int(gid))
        if name not in adapters:
            continue
        A, B, scale = adapters[name]
        upd = F.linear(F.linear(x.index_select(0, sel), A), B) * scale
        out = out.index_add(0, sel, upd)
    return out


class LoRAPool:
    """带容量上限的 LRU 适配器池（热插拔的核心）。"""

    def __init__(self, max_adapters: int = 8) -> None:
        self.max = max_adapters
        self.adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._last: Dict[str, float] = {}
        self.evicted = 0

    def add(self, name: str, A: torch.Tensor, B: torch.Tensor, scale: float) -> None:
        import time

        if name not in self.adapters and len(self.adapters) >= self.max:
            victim = min(self._last, key=self._last.get)
            self.adapters.pop(victim)
            self._last.pop(victim)
            self.evicted += 1
        self.adapters[name] = (A, B, scale)
        self._last[name] = time.time()

    def get(self, name: str):
        import time

        if name in self.adapters:
            self._last[name] = time.time()
        return self.adapters.get(name)

    def report(self) -> str:
        return f"LoRAPool: {len(self.adapters)}/{self.max}，淘汰 {self.evicted} 次"
