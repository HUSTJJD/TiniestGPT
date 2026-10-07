"""高性能内核（Triton）。

设计原则：**每个 Triton 内核都在 ``model/kernels.py`` 里有一个数值等价的
PyTorch 参考实现**，并由 ``tests/test_kernels.py`` 做一致性比对。
这样在 Windows / CPU / 无 Triton 环境下系统自动降级，功能不受影响；
在 Linux + GPU 上自动获得加速。
"""

from .triton_kernels import TRITON_AVAILABLE, register_triton_kernels, available_kernels

__all__ = ["TRITON_AVAILABLE", "register_triton_kernels", "available_kernels"]
