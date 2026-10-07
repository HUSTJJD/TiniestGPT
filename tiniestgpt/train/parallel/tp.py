"""张量并行（TP）：把矩阵乘法**按维度切开**分给多张卡。

Megatron-LM 的经典结论：一个 Transformer block 里只有两处需要通信，
而且都可以是**一次 all-reduce**：

```
        ┌── ColumnParallel（按 output 切，输出天然分片，无通信）──┐
  x ────┤                                                        ├── RowParallel（按 input 切）── all_reduce ── y
        └── ColumnParallel（同上）────────────────────────────────┘
              (attention 的 Q/K/V)                    (attention 的 O 投影)
              (FFN 的 gate/up)                        (FFN 的 down 投影)
```

为什么 TP 只能机内（NVLink）不能跨机：
每个 forward + backward 要做 **2 次 all-reduce**，通信量与 batch×seq×hidden 成正比，
PCIe/IB 的带宽根本喂不饱，所以 TP 组几乎总是"同一台机器的 8 张卡"。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .comm import DistContext

__all__ = ["ColumnParallelLinear", "RowParallelLinear", "apply_tensor_parallel",
           "TP_PATTERN", "shard_tensor"]


def shard_tensor(t: torch.Tensor, world_size: int, rank: int, dim: int) -> torch.Tensor:
    """按 ``dim`` 均分张量，返回 ``rank`` 那一片（要求能整除）。"""
    if t.shape[dim] % world_size != 0:
        raise ValueError(f"维度 {dim} 的大小 {t.shape[dim]} 不能被 world_size={world_size} 整除")
    return t.chunk(world_size, dim=dim)[rank].contiguous()


class _ParallelLinearBase(nn.Module):
    def __init__(self, ctx: DistContext, bias: bool = True) -> None:
        super().__init__()
        self.ctx = ctx

    def extra_repr(self) -> str:
        return f"world_size={self.ctx.world_size}"


class ColumnParallelLinear(_ParallelLinearBase):
    """``Y = X @ W^T``，**按 output 维度切 W**（每个 rank 算一部分输出通道）。

    :param gather_output: True 则在 forward 末尾 all-gather 回完整输出
        （最后一层或需要完整张量时打开；attention 内部通常关掉，
        让 RowParallel 直接消费分片，省一次通信）。
    """

    def __init__(self, in_features: int, out_features: int, ctx: DistContext,
                 bias: bool = True, gather_output: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(ctx, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        ws = ctx.world_size
        if out_features % ws != 0:
            raise ValueError(f"out_features={out_features} 不能被 world_size={ws} 整除")
        self.out_per_rank = out_features // ws

        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(
            torch.empty(self.out_per_rank, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 nn.Linear 相同的 Kaiming 初始化（切片前先按完整形状算bound）
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, ctx: DistContext, rank: int,
                    gather_output: bool = False) -> "ColumnParallelLinear":
        """从一个完整 ``nn.Linear`` 切出本 rank 的分片（用于模型手术与测试）。"""
        mod = cls(linear.in_features, linear.out_features, ctx,
                  bias=linear.bias is not None, gather_output=gather_output,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            mod.weight.copy_(shard_tensor(linear.weight.data, ctx.world_size, rank, 0))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(shard_tensor(linear.bias.data, ctx.world_size, rank, 0))
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)          # 输出已是分片
        if self.gather_output and self.ctx.parallel:
            y = self.ctx.all_gather(y, dim=-1)
        return y


class RowParallelLinear(_ParallelLinearBase):
    """``Y = X @ W^T``，**按 input 维度切 W**（要求输入 x 已是分片）。

    forward 末尾必须做一次 **all-reduce**（求和），把各 rank 的部分和加起来。
    bias 要放在 all-reduce **之后**加，否则会被 world_size 放大。
    """

    def __init__(self, in_features: int, out_features: int, ctx: DistContext,
                 bias: bool = True, input_is_parallel: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(ctx, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        ws = ctx.world_size
        if in_features % ws != 0:
            raise ValueError(f"in_features={in_features} 不能被 world_size={ws} 整除")
        self.in_per_rank = in_features // ws

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_rank, device=device, dtype=dtype))
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, ctx: DistContext, rank: int,
                    input_is_parallel: bool = True) -> "RowParallelLinear":
        mod = cls(linear.in_features, linear.out_features, ctx,
                  bias=linear.bias is not None, input_is_parallel=input_is_parallel,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            mod.weight.copy_(shard_tensor(linear.weight.data, ctx.world_size, rank, 1))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel and self.ctx.parallel:
            # 模拟模式下输入已经是本 rank 的分片；真实分布式下同理
            if x.shape[-1] == self.in_features:
                x = shard_tensor(x, self.ctx.world_size, self.ctx.rank, -1)
        y = F.linear(x, self.weight)                     # 不加 bias
        if self.ctx.parallel:
            y = self.ctx.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y


# --------------------------------------------------------------------------- #
# 模型手术：把普通模型改成张量并行
# --------------------------------------------------------------------------- #
# 名字后缀 → (切法, 是否 gather_output / input_is_parallel)
TP_PATTERN: dict[str, Tuple[str, bool]] = {
    "q_proj": ("column", False),     # 输出分片 = 本 rank 负责一部分注意力头
    "k_proj": ("column", False),
    "v_proj": ("column", False),
    "o_proj": ("row", True),         # 输入是分片 → all_reduce 还原
    "w_in": ("column", False),       # FFN 的 gate/up 融合矩阵
    "w_out": ("row", True),          # FFN 的 down 投影
}


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def apply_tensor_parallel(model: nn.Module, ctx: DistContext, rank: int = 0,
                          pattern: Optional[dict] = None,
                          skip: Iterable[str] = ()) -> List[str]:
    """把模型里匹配 :data:`TP_PATTERN` 的 ``nn.Linear`` 替换成并行版本。

    注意力需要同步修改"头数"：Q 按 output 切之后，本 rank 只有
    ``n_heads / world_size`` 个头，所以要把 ``n_heads`` / ``n_kv_heads`` 一起改掉
    （否则 ``view(B, T, H, D)`` 会直接崩）。

    返回被替换的模块名列表，便于日志与测试断言。
    """
    pat = pattern or TP_PATTERN
    skip = set(skip)
    replaced: List[str] = []

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf not in pat or name in skip:
            continue
        kind, flag = pat[leaf]

        parent_path = ".".join(name.split(".")[:-1])
        parent = model
        if parent_path:
            parent = model.get_submodule(parent_path)

        if kind == "column":
            new: nn.Module = ColumnParallelLinear.from_linear(mod, ctx, rank, gather_output=flag)
        else:
            new = RowParallelLinear.from_linear(mod, ctx, rank, input_is_parallel=flag)

        # 注意力头数按 world_size 缩小：q_proj → n_heads，k_proj → n_kv_heads。
        # 每个属性只能被除一次（否则 q/k 各自触发会连除两次）。
        head_attr = {"q_proj": "n_heads", "k_proj": "n_kv_heads"}.get(leaf)
        if head_attr and hasattr(parent, head_attr):
            cur = getattr(parent, head_attr)
            if cur % ctx.world_size == 0 and cur // ctx.world_size >= 1:
                setattr(parent, head_attr, cur // ctx.world_size)

        _set_module(model, name, new)
        replaced.append(name)
    return replaced
