"""手写 CUDA 内核（第一层：CUDA 编程与算子优化）。

目录::

    kernels/
    ├── csrc/           # .cu 源码，按学习顺序编号
    │   ├── common.cuh       公共宏 / include
    │   ├── 01_vector_add.cu 第一个实用 kernel
    │   ├── 02_reduce.cu     Reduce 三连（原子加 → 共享内存 → warp shuffle）
    │   ├── 03_gemm.cu       GEMM（朴素 vs 共享内存分块）
    │   ├── 04_softmax.cu    Softmax（朴素 vs online normalizer）
    │   ├── 05_transpose.cu  Bank Conflict 实验（32×32 转置 + padding）
    │   └── bindings.cu      pybind11 导出
    ├── loader.py       # JIT 编译与降级
    └── cuda_ops.py     # Python 封装 + PyTorch 参考实现 + 微基准

配套文档：``docs/08-cuda.md``。
"""

from __future__ import annotations

__all__ = ["cuda_ops", "loader", "available"]

__version__ = "0.1.0"


def available() -> bool:
    from . import cuda_ops

    return cuda_ops.available()
