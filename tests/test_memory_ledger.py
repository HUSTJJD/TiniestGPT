"""显存账本测试：把"算得清账"这条检验标准变成可执行断言。

对照 AIInfraGuide 路线第二层：
  * 7B + AdamW → 参数 14GB、梯度 14GB、优化器状态 84GB，合计 112GB
  * LLaMA-2-7B（32 层 / 32 头 / head_dim=128 / S=4096 / B=16 / fp16）→ KV Cache ≈ 32GB
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tiniestgpt.common.memory_ledger import (
    ModelShape, activation_bytes_per_token, dtype_bytes, estimate_inference_memory,
    estimate_training_memory, format_report, from_model, kv_cache_bytes,
    optimizer_state_bytes,
)

_GiB = 1024 ** 3


def _llama2_7b(batch: int = 16, seq_len: int = 4096) -> ModelShape:
    return ModelShape(
        n_params=7_000_000_000, n_layers=32, dim=4096, hidden_dim=11008,
        n_heads=32, n_kv_heads=32, head_dim=128, vocab_size=32000,
        seq_len=seq_len, batch_size=batch,
    )


def test_dtype_bytes():
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("int4") == 0.5


def test_optimizer_state_bytes():
    assert optimizer_state_bytes(1_000_000, "sgd") == 0
    assert optimizer_state_bytes(1_000_000, "momentum") == 4_000_000
    assert optimizer_state_bytes(1_000_000, "adamw") == 12_000_000
    # AdamW 的优化器状态是 fp16 参数的 6 倍 → 这就是 ZeRO-1 先砍它的原因
    assert optimizer_state_bytes(1000, "adamw") == 6 * (2 * 1000)


def test_training_memory_7b_matches_the_classic_number():
    """7B + AdamW + 混合精度：112 GB（不考虑激活）。"""
    led = estimate_training_memory(_llama2_7b(batch=1, seq_len=1), optimizer="adamw",
                                   param_dtype="bf16", grad_dtype="bf16")
    total_bytes = (led.params_mb + led.grads_mb + led.optimizer_mb) * 1024 ** 2
    assert abs(total_bytes - 16 * 7_000_000_000) / (16 * 7_000_000_000) < 1e-9
    assert abs(led.optimizer_mb * 1024 ** 2 / _GiB - 84e9 / _GiB) < 0.5


def test_zero_stages_save_memory_monotonically():
    shape = _llama2_7b(batch=1, seq_len=2048)
    ws = 8
    totals, states = [], []
    for stage in (0, 1, 2, 3):
        led = estimate_training_memory(shape, "adamw", world_size=ws, zero_stage=stage)
        totals.append(led.total_mb)
        # ZeRO 切的是"状态"（参数/梯度/优化器），**不切激活** —— 这点很容易记错
        states.append(led.params_mb + led.grads_mb + led.optimizer_mb)
    assert totals[0] > totals[1] > totals[2] > totals[3]
    assert states[0] > states[1] > states[2] > states[3]
    assert abs(states[3] / states[0] - 1 / ws) < 0.02


def test_checkpointing_trades_compute_for_memory():
    shape = _llama2_7b(batch=8, seq_len=2048)
    off = estimate_training_memory(shape, checkpointing=False)
    on = estimate_training_memory(shape, checkpointing=True)
    assert on.activations_mb < off.activations_mb * 0.5
    assert on.total_mb < off.total_mb
    assert activation_bytes_per_token(shape, checkpointing=True) < \
        activation_bytes_per_token(shape, checkpointing=False)


def test_kv_cache_llama2_7b_is_about_32gib():
    """2 · L · Hkv · D · S · B · 2B = 2·32·32·128·4096·16·2 ≈ 34.4e9 B ≈ 32 GiB。"""
    kv = kv_cache_bytes(_llama2_7b(), dtype="fp16")
    assert abs(kv - 2 * 32 * 32 * 128 * 4096 * 16 * 2) < 1
    assert abs(kv / _GiB - 32.0) < 0.5


def test_kv_cache_scales_linearly():
    a = kv_cache_bytes(_llama2_7b(seq_len=1024, batch=1))
    b = kv_cache_bytes(_llama2_7b(seq_len=2048, batch=1))
    c = kv_cache_bytes(_llama2_7b(seq_len=1024, batch=2))
    assert abs(b / a - 2.0) < 1e-9
    assert abs(c / a - 2.0) < 1e-9


def test_gqa_reduces_kv_cache():
    shape = _llama2_7b()
    mha = kv_cache_bytes(shape)
    gqa = kv_cache_bytes(ModelShape(**{**shape.__dict__, "n_kv_heads": 8}))
    assert abs(gqa / mha - 0.25) < 1e-9


def test_inference_memory_weights_plus_kv():
    led = estimate_inference_memory(_llama2_7b(batch=4, seq_len=2048), weight_dtype="fp16")
    assert led.grads_mb == 0 and led.optimizer_mb == 0
    assert abs(led.total_mb - (led.params_mb + led.kv_cache_mb)) < 1e-6


def test_from_model_matches_manual_count():
    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))
    n = sum(p.numel() for p in model.parameters())
    led = from_model(model, optimizer="adamw")
    assert abs(led.params_mb * 1024 ** 2 - n * 4) < 1            # 默认 fp32 参数
    assert abs(led.optimizer_mb * 1024 ** 2 - 12 * n) < 1


def test_format_report_runs():
    led = estimate_training_memory(_llama2_7b(batch=4, seq_len=2048))
    text = format_report(led, title="7B 训练")
    assert "7B 训练" in text and "合计" in text
