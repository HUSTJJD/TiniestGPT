"""量化工具箱：从"最省事"到"最先进"的完整谱系。

用法示例::

    from tiniestgpt.inference.quantization import quantize_model, gptq_quantize_model

    # 1) 直接 RTN（round-to-nearest）权重量化
    quantize_model(model, bits=4, group_size=128)

    # 2) 校准数据驱动的 GPTQ（更准）
    calib = collect_calibration_inputs(model, batches)     # {模块名: [激活...]}
    gptq_quantize_model(model, calib, bits=4, group_size=128)

    # 3) KV Cache 量化（decode 提速最明显）
    cache = QuantizedPagedKVCache(...)
"""

from .base import (
    QuantScheme, QuantizedLinear, quantize_model, quantize_absmax, dequantize,
    quantize_per_channel, quantize_per_token, pack_int4, unpack_int4,
    nf4_quantize, nf4_dequantize, fp8_quantize, fp8_dequantize,
)
from .gptq import gptq_quantize_linear, gptq_quantize_model
from .awq import awq_quantize_model, search_scales, pseudo_quantize
from .smoothquant import smoothquant_quantize_model, compute_smooth_scales, SmoothQuantLinear
from .kvquant import QuantizedPagedKVCache, quantize_kv, dequantize_kv
from .calibration import CalibrationCollector, collect_calibration_inputs

__all__ = [
    "QuantScheme", "QuantizedLinear", "quantize_model", "quantize_absmax", "dequantize",
    "quantize_per_channel", "quantize_per_token", "pack_int4", "unpack_int4",
    "nf4_quantize", "nf4_dequantize", "fp8_quantize", "fp8_dequantize",
    "gptq_quantize_linear", "gptq_quantize_model",
    "awq_quantize_model", "search_scales", "pseudo_quantize",
    "smoothquant_quantize_model", "compute_smooth_scales", "SmoothQuantLinear",
    "QuantizedPagedKVCache", "quantize_kv", "dequantize_kv",
    "CalibrationCollector", "collect_calibration_inputs",
]
