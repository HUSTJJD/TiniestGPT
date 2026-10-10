"""P1 训练 + 推理：FP8、分布式 Muon、Context Parallel、EMA、chunked CE、
MoE 分组 GEMM、多状态 Cache、PD 解耦、Flash-Decoding、Sleep mode。"""

import torch
import torch.nn as nn

from tiniestgpt.train.fp8 import (FP8Config, FP8Manager, dequantize_blockwise,
                                  quantize_blockwise)
from tiniestgpt.train.ema import AsyncCheckpointer, EMA, SpikeGuard
from tiniestgpt.train.losses import chunked_cross_entropy
from tiniestgpt.train.parallel.context_parallel import (ContextParallelConfig,
                                                        context_parallel_attention)
from tiniestgpt.train.parallel.muon_dist import (DistributedMuon, MuonDistConfig,
                                                 newton_schulz_dist, shard_flat)
from tiniestgpt.inference.cache_manager import CacheKind, HierarchicalCache, MultiStateCache
from tiniestgpt.inference.disaggregated import DisaggregatedServer, PDConfig
from tiniestgpt.inference.kernels.flash_decoding import flash_decoding, recommend_split
from tiniestgpt.inference.kernels.moe_gemm import (ExpertParallel, load_balance_report,
                                                   moe_grouped_forward)
from tiniestgpt.inference.sleep import SleepLevel, SleepManager, weight_bytes


# --------------------------------------------------------------------------- #
#  训练
# --------------------------------------------------------------------------- #
def test_fp8_blockwise_roundtrip_error_is_small():
    w = torch.randn(128, 64)
    q, s = quantize_blockwise(w, 32)
    d = dequantize_blockwise(q, s, 32, w.shape)
    rel = float((d - w).abs().mean() / w.abs().mean())
    assert rel < 0.05, f"E4M3 分块量化误差过大: {rel}"


def test_fp8_manager_keeps_master_weights():
    m = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))
    before = [p.detach().clone() for p in m.parameters()]
    mgr = FP8Manager(FP8Config(block=16))
    assert mgr.apply(m) >= 1
    m(torch.randn(2, 32))
    for a, b in zip(before, m.parameters()):
        assert torch.equal(a, b), "FP8 不应修改主权重对象"


def test_distributed_muon_updates_only_own_shard():
    torch.manual_seed(0)
    m = nn.Linear(16, 16)
    w0 = m.weight.detach().clone()
    opt = DistributedMuon(m.parameters(), MuonDistConfig(world_size=2, rank=0, lr=0.1))
    m(torch.randn(2, 16)).sum().backward()
    opt.step()
    flat = m.weight.detach().reshape(-1)
    changed = (flat != w0.reshape(-1)).sum()
    assert 0 < changed <= flat.numel() // 2 + 16, "rank0 只应更新自己那一片"


def test_newton_schulz_makes_spectrum_more_uniform():
    """Newton-Schulz 是**近似**正交化：它把奇异值往一起推，但不会精确等于 1。

    所以可检验的性质是"谱更均匀"（min/max 比值显著提高），
    而不是"XᵀX == I"——后者在 5~6 步迭代下根本不成立。
    """
    torch.manual_seed(0)
    G = torch.randn(32, 16)
    s0 = torch.linalg.svdvals(G)
    s1 = torch.linalg.svdvals(newton_schulz_dist(G, steps=6))
    r0, r1 = float(s0.min() / s0.max()), float(s1.min() / s1.max())
    assert r1 > r0, f"奇异值没有变得更均匀: {r0} -> {r1}"
    assert r1 > 0.4, f"正交化效果太弱: {r1}"


def test_context_parallel_matches_full_attention():
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 24, 8
    q, k, v = (torch.randn(B, L, H, D) for _ in range(3))
    full = context_parallel_attention(q, k, v, ContextParallelConfig(world_size=1))
    for ws in (2, 3, 4):
        got = context_parallel_attention(q, k, v, ContextParallelConfig(world_size=ws))
        assert torch.allclose(full, got, atol=1e-5), f"world_size={ws} 不等价"


def test_ema_apply_and_restore():
    m = nn.Linear(8, 8)
    e = EMA(m, decay=0.9)
    e.update(m)                                   # shadow ← 当前权重
    shadow = e.shadow["weight"].clone()
    changed = m.weight.detach().clone() + 1.0
    m.weight.data.copy_(changed)
    e.apply_to(m)
    assert torch.allclose(m.weight.detach(), shadow), "apply_to 应把 EMA 权重装上"
    e.restore(m)
    assert torch.allclose(m.weight.detach(), changed), "restore 应还原原权重"


def test_spike_guard_and_async_checkpoint(tmp_path):
    # 需要攒够窗口长度的一半（默认 max(window//2, 5)）才开始判定
    g = SpikeGuard(factor=2.0, window=5, cooldown=2)
    assert not any(g.observe(1.0) for _ in range(5))
    assert g.observe(5.0) is True
    assert "触发" in g.report()

    ck = AsyncCheckpointer()
    assert ck.save({"a": torch.ones(3)}, str(tmp_path / "x.pt"))
    ck.wait()
    assert ck.saved == 1


def test_chunked_ce_matches_full():
    torch.manual_seed(0)
    lg = torch.randn(3, 20, 40)
    lb = torch.randint(0, 40, (3, 20))
    a = chunked_cross_entropy(lg, lb, chunk=0)
    b = chunked_cross_entropy(lg, lb, chunk=7)
    assert torch.allclose(a, b, atol=1e-5)


# --------------------------------------------------------------------------- #
#  推理
# --------------------------------------------------------------------------- #
def test_moe_grouped_forward_and_balance():
    torch.manual_seed(0)
    N, C, E, H = 24, 16, 4, 32
    out, idx = moe_grouped_forward(torch.randn(N, C), torch.randn(E, C),
                                   torch.randn(E, H, C), torch.randn(E, C, H), top_k=2)
    assert out.shape == (N, C)
    rep = load_balance_report(idx, E)
    assert rep["mean_load"] == float(N * 2 / E)
    assert rep["imbalance"] >= 1.0


def test_expert_parallel_comm_scales_with_hidden():
    ep = ExpertParallel(world_size=8)
    b1 = ep.all_to_all(1024, 8192, 8)
    b2 = ep.all_to_all(1024, 2048, 8)      # LatentMoE：降维后 payload 变 1/4
    assert abs(b1 / b2 - 4.0) < 1e-6


def test_multi_state_cache_and_sharing():
    msc = MultiStateCache(4, [CacheKind.PAGED, CacheKind.RECURRENT,
                              CacheKind.RING, CacheKind.PAGED],
                          n_kv_heads=2, head_dim=8, window=16, recurrent_shape=(4, 8, 8))
    assert msc.slot_count() == 4
    msc.bind_shared(3, 0)
    assert msc.slot_count() == 3
    assert "MultiStateCache" in msc.report()


def test_hierarchical_cache_offload_and_fetch():
    hc = HierarchicalCache()
    hc.put("a", torch.zeros(1000))
    assert hc.offload("a")
    assert hc.stats["offload"] == 1
    t = hc.fetch("a")
    assert t is not None and hc.stats["fetch"] == 1
    assert hc.fetch("missing") is None and hc.stats["miss"] == 1


def test_flash_decoding_matches_dense():
    torch.manual_seed(0)
    B, H, L, D = 2, 4, 48, 8
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, D)
    s = torch.matmul(q, k.transpose(1, 2).transpose(-1, -2)) / (D ** 0.5)
    ref = torch.matmul(torch.softmax(s, -1), v.transpose(1, 2))
    from tiniestgpt.inference.kernels.flash_decoding import SplitKVConfig
    got = flash_decoding(q, k, v, SplitKVConfig(n_split=6, min_chunk=4, causal=False),
                         auto=False)
    assert torch.allclose(ref, got, atol=1e-5)
    assert recommend_split(4096) > 1


def test_pd_disaggregated_server():
    """PD 解耦的 worker 需要真模型（要 kv_cache_bytes_per_token）。"""
    from tiniestgpt.model.factory import build_model

    m = build_model("nano")
    srv = DisaggregatedServer(m, PDConfig(n_prefill=2, n_decode=1))
    r = srv.handle(torch.randint(0, m.cfg.vocab_size, (1, 8)))
    assert r["kv_bytes"] > 0
    assert "PD 解耦" in srv.report()


def test_sleep_mode_unloads_and_restores():
    m = nn.Linear(64, 64)
    before = m.weight.detach().clone()
    sm = SleepManager(m)
    n = weight_bytes(m)
    assert n > 0
    sm.sleep(SleepLevel.UNLOAD_WEIGHTS)
    assert m.weight.numel() == 0
    sm.wake_up()
    assert torch.allclose(m.weight.detach(), before)
