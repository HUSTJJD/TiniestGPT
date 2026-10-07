"""推理层：AI Infra 的全部内容都在这里。

模块地图（建议按这个顺序阅读）::

    kv_cache（在 model/ 下）  →  scheduler.py      连续批处理 / 分页 / prefix caching
                             →  engine.py         组装成系统
    sampler.py                  采样与解码策略
    quantization/               权重量化 + KV 量化
    kernels/triton_kernels.py   高性能内核（自动降级）
    speculative.py              投机解码
    cuda_graph.py               CUDA Graph 捕获与 replay
    server.py                   OpenAI 兼容服务
"""

from .engine import EngineConfig, InferenceEngine, RequestOutput, build_engine
from .sampler import Sampler, SamplingParams
from .scheduler import Scheduler, Sequence, SequenceStatus
from .speculative import SpeculativeDecoder

__all__ = [
    "EngineConfig", "InferenceEngine", "RequestOutput", "build_engine",
    "Sampler", "SamplingParams",
    "Scheduler", "Sequence", "SequenceStatus",
    "SpeculativeDecoder",
]
