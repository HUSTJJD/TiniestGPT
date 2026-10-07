"""并行策略（TP / SP / PP / ZeRO）的数值一致性测试。

关键设计：所有用例都在 **world_size=1**（退化为普通实现）与
**world_size=2 的模拟模式**（单进程依次扮演两张卡）两种情况下验证，
前者保证"接入不改变语义"，后者保证"切分与通信是对的"。
因此**不需要多卡**也能跑。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tiniestgpt.train.parallel import (
    DistContext, PipelineParallel, RowParallelLinear, ZeroOptimizer, apply_tensor_parallel,
    bubble_ratio, gather_from_sp, make_stages, partition_params, run_ranks, scatter_to_sp,
    zero_memory_report,
)
from tiniestgpt.train.parallel.tp import ColumnParallelLinear


def _tiny_transformer():
    from tiniestgpt.model.config import ModelConfig
    from tiniestgpt.model.transformer import Transformer

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, attn_type="gqa")
    return Transformer(cfg)


# --------------------------------------------------------------------------- #
# 张量并行
# --------------------------------------------------------------------------- #
def test_column_parallel_world_size_1_matches_linear():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    ctx = DistContext(world_size=1)
    col = ColumnParallelLinear.from_linear(lin, ctx, rank=0)
    assert torch.allclose(col(x), lin(x), atol=1e-6)


def test_row_parallel_world_size_1_matches_linear():
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    ctx = DistContext(world_size=1)
    row = RowParallelLinear.from_linear(lin, ctx, rank=0)
    assert torch.allclose(row(x), lin(x), atol=1e-6)


def test_column_parallel_simulated_two_ranks():
    """两个 rank 各自算一半输出通道，all-gather 后 == 完整 Linear。"""
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    ctx = DistContext(world_size=2)

    def run(rank: int) -> torch.Tensor:
        ctx.rank = rank
        return ColumnParallelLinear.from_linear(lin, ctx, rank, gather_output=True)(x)

    got = run_ranks(ctx, run)
    assert torch.allclose(got, lin(x), atol=1e-5)


def test_row_parallel_simulated_two_ranks():
    """输入与权重都按 input 维切开，两个 rank 的部分和 all-reduce 后 == 完整结果。"""
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    ctx = DistContext(world_size=2)

    def run(rank: int) -> torch.Tensor:
        ctx.rank = rank
        x_shard = x.chunk(2, dim=-1)[rank].contiguous()
        return RowParallelLinear.from_linear(lin, ctx, rank)(x_shard)

    got = run_ranks(ctx, run)
    assert torch.allclose(got, lin(x), atol=1e-5)


def test_apply_tensor_parallel_world_size_1_is_identical():
    """接入 TP（world_size=1）后模型输出必须与原来一模一样。"""
    model = _tiny_transformer()
    model.eval()
    x = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        ref = model(x)

    ctx = DistContext(world_size=1)
    replaced = apply_tensor_parallel(model, ctx, rank=0)
    assert replaced, "应当至少替换了几个 Linear"
    model.eval()
    with torch.no_grad():
        got = model(x)
    assert torch.allclose(got, ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# 序列并行
# --------------------------------------------------------------------------- #
def test_sequence_parallel_roundtrip_world_size_1():
    ctx = DistContext(world_size=1)
    x = torch.randn(2, 8, 16)
    assert torch.allclose(gather_from_sp(scatter_to_sp(x, ctx), ctx), x)


def test_sequence_parallel_roundtrip_simulated():
    """模拟模式下 gather 拿不到"另一张卡"的数据，只能验证形状与本 rank 分片。"""
    ctx = DistContext(world_size=2)
    x = torch.randn(2, 8, 16)
    shard = scatter_to_sp(x, ctx)
    assert shard.shape == (2, 4, 16)
    assert torch.allclose(shard, x.chunk(2, dim=1)[0])
    full = gather_from_sp(shard, ctx)
    assert full.shape == x.shape
    assert torch.allclose(full[:, :4], x[:, :4])


# --------------------------------------------------------------------------- #
# 流水线并行
# --------------------------------------------------------------------------- #
def test_pipeline_matches_sequential():
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(6)])
    x = torch.randn(8, 16)

    ref = x
    for layer in layers:
        ref = layer(ref)

    pipe = PipelineParallel(layers, num_stages=3, num_microbatches=4,
                            ctx=DistContext(world_size=1))
    got = pipe(x)
    assert torch.allclose(got, ref, atol=1e-6)


def test_make_stages_balanced():
    layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(7)])
    stages = make_stages(layers, 3)
    assert len(stages) == 3
    assert sum(len(s) for s in stages) == 7


def test_bubble_ratio_decreases_with_microbatches():
    assert bubble_ratio(1, 4) == 0.0
    assert bubble_ratio(4, 4) > bubble_ratio(4, 16)
    assert abs(bubble_ratio(4, 8) - 3 / 8) < 1e-9


# --------------------------------------------------------------------------- #
# ZeRO
# --------------------------------------------------------------------------- #
def _train_tiny(steps: int = 5):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    data = torch.randn(16, 8)
    target = torch.randn(16, 4)
    return model, data, target


def test_zero_stage1_equals_adamw_when_world_size_1():
    m1, x, y = _train_tiny()
    m2, _, _ = _train_tiny()
    opt_ref = torch.optim.AdamW(m2.parameters(), lr=0.1)
    zero = ZeroOptimizer(m1.parameters(), DistContext(world_size=1), stage=1, lr=0.1)

    for _ in range(5):
        torch.nn.functional.mse_loss(m1(x), y).backward()
        zero.step()
        zero.zero_grad()
        torch.nn.functional.mse_loss(m2(x), y).backward()
        opt_ref.step()
        opt_ref.zero_grad()

    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_zero_stage2_equals_adamw_when_world_size_1():
    m1, x, y = _train_tiny()
    m2, _, _ = _train_tiny()
    opt_ref = torch.optim.AdamW(m2.parameters(), lr=0.1)
    zero = ZeroOptimizer(m1.parameters(), DistContext(world_size=1), stage=2, lr=0.1)
    for _ in range(5):
        torch.nn.functional.mse_loss(m1(x), y).backward()
        zero.step()
        zero.zero_grad()
        torch.nn.functional.mse_loss(m2(x), y).backward()
        opt_ref.step()
        opt_ref.zero_grad()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_zero_partition_is_disjoint():
    params = [nn.Parameter(torch.zeros(i + 1)) for i in range(7)]
    owned0, other0 = partition_params(params, 2, 0)
    owned1, other1 = partition_params(params, 2, 1)
    assert len(owned0) + len(owned1) == len(params)
    assert not (set(map(id, owned0)) & set(map(id, owned1)))
    assert set(map(id, owned0)) == set(map(id, other1))


def test_zero_memory_report_monotonic():
    n = 7_000_000_000
    r0 = zero_memory_report(n, world_size=8, stage=0)
    r1 = zero_memory_report(n, world_size=8, stage=1)
    r2 = zero_memory_report(n, world_size=8, stage=2)
    r3 = zero_memory_report(n, world_size=8, stage=3)
    assert r0["per_rank_gb"] > r1["per_rank_gb"] > r2["per_rank_gb"] > r3["per_rank_gb"]
    # 7B + AdamW：16 字节/参数 = 112 GB（十进制）≈ 104.3 GiB
    assert abs(r0["no_sharding_gb"] * 1024 ** 3 - 16 * n) / (16 * n) < 1e-9
