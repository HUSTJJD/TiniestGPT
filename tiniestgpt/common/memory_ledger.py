"""显存账本：把"训练一个模型到底要多少显存"算清楚。

AI Infra 路线里反复强调的一件事：**先算账，再动手**。
拿到一个模型配置，能口算出参数/梯度/优化器状态/激活/KV Cache 各占多少，
才知道该用 DDP 还是 ZeRO、要不要重计算、能不能上长上下文。

公式（单位：字节）

* 参数（训练时计算用 bf16/fp16）      : ``2 · N``
* 梯度                                : ``2 · N``
* AdamW 优化器状态                     : ``12 · N``（fp32 主权重 + m + v）
  - SGD：``0``   Momentum-SGD：``4 · N``
* 激活（估算，见 :func:`activation_bytes_per_token`）: ``L · B · S · bytes_per_token``
* KV Cache                            : ``2 · L · Hkv · D · S · B · dtype``

经典结论：AdamW 下**优化器状态是参数的 6 倍**，
所以 ZeRO 第一刀必须砍在优化器状态上（这就是 ZeRO-1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = [
    "ModelShape", "MemoryLedger", "dtype_bytes", "optimizer_state_bytes",
    "activation_bytes_per_token", "kv_cache_bytes", "estimate_training_memory",
    "estimate_inference_memory", "from_model", "format_report",
]

_GB = 1024 ** 3
_MB = 1024 ** 2

_DTYPE_BYTES = {
    "fp32": 4, "float32": 4, "torch.float32": 4,
    "fp16": 2, "float16": 2, "torch.float16": 2,
    "bf16": 2, "bfloat16": 2, "torch.bfloat16": 2,
    "fp8": 1, "float8": 1,
    "int8": 1, "int4": 0.5,
}

# 每个参数需要多少字节的**优化器状态**（fp32 主权重 + 动量）
_OPTIMIZER_BYTES = {
    "sgd": 0.0,
    "momentum": 4.0,          # 一阶动量
    "adam": 12.0,             # fp32 主权重 + m + v
    "adamw": 12.0,
    "lion": 4.0,              # 只存一阶动量
    "muon": 4.0,              # 近似：一阶动量
    "sophia": 8.0,            # 一阶动量 + 二阶估计（Hessian 对角近似，可省）
    "adafactor": 0.5,         # 低秩分解，量级极小
}


def dtype_bytes(name: str) -> float:
    try:
        return _DTYPE_BYTES[str(name).lower()]
    except KeyError:
        raise ValueError(f"未知 dtype: {name}（可选: {sorted(set(_DTYPE_BYTES))}）")


def optimizer_state_bytes(n_params: int, kind: str = "adamw") -> float:
    """优化器状态占用的字节数。"""
    per = _OPTIMIZER_BYTES.get(kind.lower())
    if per is None:
        raise ValueError(f"未知优化器: {kind}（可选: {sorted(_OPTIMIZER_BYTES)}）")
    return n_params * per


@dataclass
class ModelShape:
    """算账所需的全部形状信息。"""
    n_params: int
    n_layers: int = 1
    dim: int = 0
    hidden_dim: int = 0           # FFN 中间维
    n_heads: int = 0
    n_kv_heads: int = 0           # GQA/MQA 下 < n_heads
    head_dim: int = 0
    vocab_size: int = 0
    seq_len: int = 2048
    batch_size: int = 1
    active_params: Optional[int] = None      # MoE：激活的参数量
    n_experts: int = 1
    n_active_experts: int = 1

    @property
    def effective_params(self) -> int:
        return self.active_params or self.n_params


def activation_bytes_per_token(shape: ModelShape, param_dtype: str = "bf16",
                               checkpointing: bool = False) -> float:
    """每个 token 每层需要保存多少**激活**字节（量级估计，±30%）。

    不重计算时，每层每 token 要存（元素个数）：

    * LayerNorm / 残差 / dropout 掩码   ≈ ``6·d``
    * Q/K/V 与 attention 输出           ≈ ``d + 2·Hkv·D``
    * FFN 的 gate / up / 乘积           ≈ ``3·d_ff``
    * **attention 分数矩阵**            ≈ ``H·S`` ← 长序列下的绝对大头

    最后一项就是 FlashAttention 存在的理由：它把 ``H·S`` 这一项拿掉了。

    开了重计算（Activation Checkpointing）后只保留每层输入（``d`` 个元素），
    反向时现算——**这就是"用计算换显存"**。
    """
    if shape.n_layers <= 0:
        return 0.0
    hkv = shape.n_kv_heads or shape.n_heads
    if checkpointing:
        elems = float(shape.dim)                      # 只存每层输入
    else:
        elems = (6.0 * shape.dim
                 + shape.dim + 2.0 * hkv * shape.head_dim
                 + 3.0 * shape.hidden_dim
                 + float(shape.n_heads * shape.seq_len))
    return elems * dtype_bytes(param_dtype)


def kv_cache_bytes(shape: ModelShape, dtype: str = "bf16",
                   seq_len: Optional[int] = None, batch: Optional[int] = None) -> float:
    """KV Cache 总字节数：``2 · L · Hkv · D · S · B · dtype``。"""
    s = seq_len or shape.seq_len
    b = batch or shape.batch_size
    return 2.0 * shape.n_layers * shape.n_kv_heads * shape.head_dim * s * b * dtype_bytes(dtype)


@dataclass
class MemoryLedger:
    params_mb: float = 0.0
    grads_mb: float = 0.0
    optimizer_mb: float = 0.0
    activations_mb: float = 0.0
    kv_cache_mb: float = 0.0
    total_mb: float = 0.0
    notes: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "params_mb": self.params_mb,
            "grads_mb": self.grads_mb,
            "optimizer_mb": self.optimizer_mb,
            "activations_mb": self.activations_mb,
            "kv_cache_mb": self.kv_cache_mb,
            "total_mb": self.total_mb,
        }


def estimate_training_memory(shape: ModelShape, optimizer: str = "adamw",
                             param_dtype: str = "bf16", grad_dtype: str = "bf16",
                             master_fp32: bool = True, checkpointing: bool = False,
                             world_size: int = 1, zero_stage: int = 0,
                             tp_size: int = 1) -> MemoryLedger:
    """估算训练显存（**每卡**）。

    :param zero_stage: 0=纯 DDP；1=切优化器状态；2=再切梯度；3=参数也切
    :param tp_size:    张量并行度（参数/梯度/激活都按 1/tp 计）
    """
    n = shape.effective_params
    pb, gb = dtype_bytes(param_dtype), dtype_bytes(grad_dtype)
    master = 4.0 if master_fp32 else 0.0

    params = n * pb
    grads = n * gb
    optim = optimizer_state_bytes(n, optimizer)
    if not master_fp32:                     # 不保留 fp32 主权重就省 4N
        optim = max(optim - n * 4.0, 0.0)

    act = (activation_bytes_per_token(shape, param_dtype, checkpointing)
           * shape.batch_size * shape.seq_len * max(shape.n_layers, 1))

    # ---- 并行切分 ----
    if world_size > 1:
        if zero_stage >= 1:
            optim /= world_size
        if zero_stage >= 2:
            grads /= world_size
        if zero_stage >= 3:
            params /= world_size
    if tp_size > 1:
        params /= tp_size
        grads /= tp_size
        optim /= tp_size
        act /= tp_size

    total = params + grads + optim + act
    return MemoryLedger(
        params_mb=params / _MB, grads_mb=grads / _MB, optimizer_mb=optim / _MB,
        activations_mb=act / _MB, kv_cache_mb=0.0, total_mb=total / _MB,
        notes={"optimizer": optimizer, "master_fp32": float(master_fp32),
               "checkpointing": float(checkpointing), "world_size": float(world_size),
               "zero_stage": float(zero_stage), "tp_size": float(tp_size)},
    )


def estimate_inference_memory(shape: ModelShape, weight_dtype: str = "bf16",
                              kv_dtype: str = "bf16", seq_len: Optional[int] = None,
                              batch: Optional[int] = None) -> MemoryLedger:
    """估算推理显存：权重 + KV Cache（没有梯度与优化器状态）。"""
    weights = shape.n_params * dtype_bytes(weight_dtype)
    kv = kv_cache_bytes(shape, kv_dtype, seq_len, batch)
    return MemoryLedger(
        params_mb=weights / _MB, kv_cache_mb=kv / _MB,
        total_mb=(weights + kv) / _MB,
        notes={"weight_dtype": weight_dtype, "kv_dtype": kv_dtype},
    )


def from_model(model: "object", optimizer: Optional[str] = None,
               param_dtype: str = "bf16") -> MemoryLedger:
    """用真实模型统计参数/显存（需要与 ``estimate_*`` 对账）。"""
    import torch.nn as nn

    assert isinstance(model, nn.Module)
    params = grads = 0.0
    for p in model.parameters():
        params += p.numel() * p.element_size()
        if p.requires_grad:
            grads += p.numel() * p.element_size()
    optim = optimizer_state_bytes(
        sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer or "sgd")
    return MemoryLedger(params_mb=params / _MB, grads_mb=grads / _MB,
                        optimizer_mb=optim / _MB, total_mb=(params + grads + optim) / _MB)


def format_report(ledger: MemoryLedger, title: str = "显存账本") -> str:
    rows = [
        ("参数", ledger.params_mb),
        ("梯度", ledger.grads_mb),
        ("优化器状态", ledger.optimizer_mb),
        ("激活", ledger.activations_mb),
        ("KV Cache", ledger.kv_cache_mb),
    ]
    lines = [f"=== {title} ==="]
    for name, mb in rows:
        if mb <= 0:
            continue
        lines.append(f"  {name:<12} {mb:10.1f} MB   ({mb / 1024:6.2f} GB)")
    lines.append(f"  {'合计':<12} {ledger.total_mb:10.1f} MB   ({ledger.total_mb / 1024:6.2f} GB)")
    return "\n".join(lines)
