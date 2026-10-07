"""并行策略：**张量并行 / 序列并行 / 流水线并行 / ZeRO**。

对应 AIInfraGuide 路线第二层。目录::

    parallel/
    ├── comm.py   通信抽象（真实分布式 + 单机模拟多卡）
    ├── tp.py     ColumnParallelLinear / RowParallelLinear / 模型手术
    ├── sp.py     序列并行：进出 TP 区的 scatter / gather
    ├── pp.py     流水线切分与 1F1B 调度、气泡分析
    └── zero.py   ZeRO-1/2/3 手写实现 + 显存收益估算

设计原则（与全项目一致）：**没有多卡也能学**。
所有模块在 ``world_size=1`` 时数值上退化为普通实现，
``comm.DistContext`` 提供"单进程扮演 N 张卡"的模拟模式，
因此在笔记本上就能验证 3D 并行的数学正确性。
"""

from __future__ import annotations

from .comm import DistContext, run_ranks
from .pp import PipelineParallel, bubble_ratio, make_stages
from .sp import gather_from_sp, reduce_scatter_to_sp, scatter_to_sp
from .tp import ColumnParallelLinear, RowParallelLinear, apply_tensor_parallel
from .zero import ZeroOptimizer, partition_params, zero_memory_report

__all__ = [
    "DistContext", "run_ranks",
    "ColumnParallelLinear", "RowParallelLinear", "apply_tensor_parallel",
    "scatter_to_sp", "gather_from_sp", "reduce_scatter_to_sp",
    "PipelineParallel", "bubble_ratio", "make_stages",
    "ZeroOptimizer", "partition_params", "zero_memory_report",
]
