"""模型架构层：一个"小到能在 3060 上训、先进到对齐 2026 年 SOTA"的 Transformer。

每一层的实现都遵循同一条原则：**先写可读的 reference，再给出可插拔的高性能后端**。
"""

from .config import ModelConfig, PRESETS
from .transformer import Transformer
from .kv_cache import CacheView, DenseKVCache, PagedKVCache, KVCache
from .factory import build_model

__all__ = [
    "ModelConfig", "PRESETS", "Transformer",
    "CacheView", "DenseKVCache", "PagedKVCache", "KVCache",
    "build_model",
]
