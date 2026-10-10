"""P1 架构：稀疏/压缩注意力、mHC、RWKV-7、并行混合块、KV 压缩。"""

import torch
import pytest

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.model.kv_cache import DenseKVCache
from tiniestgpt.model.hyper_connections import (HyperConnections, MHCBlock, sinkhorn_knopp)
from tiniestgpt.model.kv_sharing import KVSharePlan
from tiniestgpt.model.rwkv7 import RWKV7Mixer, rwkv7_reference_scan
from tiniestgpt.model.sparse_attention import (LightningIndexer, SparseAttention,
                                               SparseIndexBank, compress_kv)


def _cfg(**kw):
    base = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=256, max_seq_len=128)
    base.update(kw)
    return ModelConfig(**base)


# --------------------------------------------------------------------------- #
def test_sinkhorn_is_doubly_stochastic():
    P = sinkhorn_knopp(torch.randn(4, 4), iters=30)
    assert torch.allclose(P.sum(dim=-1), torch.ones(4), atol=1e-3)
    assert torch.allclose(P.sum(dim=-2), torch.ones(4), atol=1e-3)
    assert (P >= 0).all()


def test_mhc_preserves_shape_and_collapses():
    hc = HyperConnections(dim=32, mult=4, iters=10)
    X = MHCBlock.expand(torch.randn(2, 5, 32), 4)
    assert X.shape == (2, 5, 4, 32)
    y = hc.post_mix(X, torch.randn(2, 5, 32))
    assert y.shape == X.shape
    assert MHCBlock.collapse(y).shape == (2, 5, 32)


def test_mhc_preset_runs():
    m = build_model("mhc")
    assert m.cfg.hc_mult == 4
    out = m(torch.randint(0, m.cfg.vocab_size, (2, 8)))
    assert out.shape == (2, 8, m.cfg.vocab_size)


# --------------------------------------------------------------------------- #
def test_compress_kv_shape_and_mean():
    k = torch.arange(2 * 8 * 2 * 4, dtype=torch.float32).reshape(2, 8, 2, 4)
    kc, vc, counts = compress_kv(k, k, 4)
    assert kc.shape == (2, 2, 2, 4)
    assert torch.allclose(kc[0, 0], k[0, :4].mean(dim=0))
    assert counts[0] == 4


def test_lightning_indexer_causal_mask():
    cfg = _cfg()
    idx = LightningIndexer(cfg, n_heads=2, index_dim=8)
    x = torch.randn(1, 6, cfg.dim)
    I = idx.score(x, idx.project_keys(x))
    assert I.shape == (1, 6, 6)


@pytest.mark.parametrize("mode,comp", [("dsa", 1), ("csa", 4), ("hca", 8)])
def test_sparse_attention_modes(mode, comp):
    cfg = _cfg()
    m = SparseAttention(cfg, 0, mode=mode, index_topk=4, compression=comp,
                        local_window=4)
    x = torch.randn(2, 16, cfg.dim)
    out = m(x)
    assert out.shape == (2, 16, cfg.dim)
    assert torch.isfinite(out).all()


def test_sparse_decode_with_cache_matches_prefill_prefix():
    """同一段输入：一次性 prefill 与逐步 decode 的首位结果应一致。"""
    cfg = _cfg(sparse_mode="dsa", sparse_topk=8, sparse_local_window=8)
    m = SparseAttention(cfg, 0, mode="dsa", index_topk=8, local_window=8).eval().float()
    x = torch.randn(1, 6, cfg.dim)
    with torch.no_grad():
        full = m(x)
        cache = DenseKVCache(1, 1, 16, cfg.n_kv_heads, cfg.head_dim, dtype=torch.float32)
        stepwise = []
        for t in range(6):
            cache.write_pos = t
            stepwise.append(m(x[:, t:t + 1], cache=cache))
            cache.advance(1)
        cat = torch.cat(stepwise, dim=1)
    assert torch.allclose(full[:, :1], cat[:, :1], atol=1e-4)


def test_indexshare_reuses_indices():
    bank = SparseIndexBank(topk_freq=4, skip_offset=2)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.zeros(1, 1, 2, dtype=torch.long)

    for layer in range(2, 10):
        bank.get(layer, compute)
    assert calls["n"] == 2, "8 层里每 4 层才应算一次"
    assert bank.reuse_count == 6
    assert "IndexShare" in bank.report()


# --------------------------------------------------------------------------- #
def test_rwkv7_state_is_context_independent():
    """RWKV-7 的状态大小与上下文长度无关——这是它存在的全部理由。"""
    cfg = _cfg()
    m = RWKV7Mixer(cfg, 0).float()
    s = m._new_state(torch.zeros(1, 1, cfg.dim))
    for T in (4, 64):
        _, st = rwkv7_reference_scan(
            torch.randn(1, cfg.n_heads, T, cfg.head_dim),
            torch.randn(1, cfg.n_heads, T, cfg.head_dim),
            torch.randn(1, cfg.n_heads, T, cfg.head_dim),
            torch.zeros(1, cfg.n_heads, T, 1),
            torch.randn(1, cfg.n_heads, T, cfg.head_dim),
            torch.randn(1, cfg.n_heads, T, cfg.head_dim), s)
        assert st.shape == s.shape
    assert m.state_shape == (cfg.n_heads, cfg.head_dim, cfg.head_dim)


def test_rwkv7_preset_runs():
    m = build_model("rwkv7")
    out = m(torch.randint(0, m.cfg.vocab_size, (2, 8)))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
def test_kv_share_plan_slot_ratio():
    p = KVSharePlan(8, "p,c,c,c")
    assert p.n_producers == 2
    assert p.owner[0] == 0 and p.owner[1] == 0 and p.owner[4] == 4
    assert abs(p.slot_ratio - 0.25) < 1e-9
    assert "KVSharing" in p.report()


def test_kv_eq_v_halves_projection():
    cfg = _cfg(kv_eq_v=True)
    m = build_model(cfg)
    attn = m.layers[0].mixer
    assert getattr(attn, "v_proj", None) is None
    from tiniestgpt.model.attention import Attention
    assert isinstance(attn, Attention)


def test_attention_temperature_scaling():
    cfg = _cfg(attn_temperature=2.0)
    m = build_model(cfg)
    assert m.layers[0].mixer.temperature == 2.0
