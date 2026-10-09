"""2026 线性序列混合器：Gated DeltaNet / Mamba-2(SSD) 的三形态一致性。

这三种形态必须**数值等价**，否则推理与训练就会悄悄不一致——
这是线性注意力最常见的线上事故（decode 结果与 prefill 对不上）。
"""

import torch
import pytest

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.model.kv_cache import DenseKVCache
from tiniestgpt.model.sequence_mixer import GatedDeltaNet, Mamba2Mixer


def _cfg(**kw):
    base = dict(dim=64, n_layers=1, n_heads=4, n_kv_heads=2, vocab_size=256)
    base.update(kw)
    return ModelConfig(**base)


def _inputs(cfg, T=20, B=2):
    torch.manual_seed(0)
    return torch.randn(B, T, cfg.dim)


# --------------------------------------------------------------------------- #
def test_gdn_three_forms_equivalent():
    cfg = _cfg(gdn_chunk_size=8)
    m = GatedDeltaNet(cfg, 0).float()
    x = _inputs(cfg)
    q, k, v, beta, alpha = m._project(x)

    o_seq, s_seq = m.forward_sequential(q, k, v, beta, alpha)
    o_chunk, s_chunk = m.forward_chunk(q, k, v, beta, alpha)
    assert torch.allclose(o_seq, o_chunk, atol=1e-6), "GDN: sequential ≠ chunk"
    assert torch.allclose(s_seq, s_chunk, atol=1e-6), "GDN: 末状态不一致"

    state = m._new_state(q)
    outs = []
    for t in range(q.shape[2]):
        o, state = m.forward_recurrent(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
                                       beta[:, :, t:t + 1], alpha[:, :, t:t + 1], state)
        outs.append(o[:, :, 0])
    o_rec = torch.stack(outs, dim=2)
    assert torch.allclose(o_seq, o_rec, atol=1e-6), "GDN: sequential ≠ recurrent(decode)"
    assert torch.allclose(s_seq, state, atol=1e-6)


def test_gdn_delta_rule_rewrites_old_association():
    """Delta Rule 的核心：同一个 key 再次出现时改写旧关联，而不是继续累加。"""
    cfg = _cfg(gdn_chunk_size=0)
    m = GatedDeltaNet(cfg, 0).float()
    B, H, T, D = 1, cfg.n_heads, 3, cfg.head_dim
    q = torch.zeros(B, H, T, D)
    k = torch.zeros(B, H, T, D)
    v = torch.zeros(B, H, T, D)
    k[:, :, 0, 0] = 1.0      # 位置 0 与 2 用同一个 key
    k[:, :, 2, 0] = 1.0
    v[:, :, 0, 1] = 1.0      # 第一次写入 v=(0,1,...)
    v[:, :, 2, 1] = 5.0      # 第二次写入 v=(0,5,...)
    alpha = torch.ones(B, H, T)
    beta = torch.ones(B, H, T)

    o, S = m.forward_sequential(q, k, v, beta, alpha)
    # S k 应该≈第二次写入的 v（被改写），而不是两次之和
    read = torch.einsum("bhde,bhe->bhd", S, k[:, :, 2])
    assert torch.allclose(read[..., 1], torch.full_like(read[..., 1], 5.0), atol=1e-4)


def test_mamba2_three_forms_equivalent():
    cfg = _cfg(ssm_state_size=16, ssm_chunk_size=8)
    m = Mamba2Mixer(cfg, 0).float()
    x = _inputs(cfg)
    x_, b_, c_, a_, g_ = m._split(x)

    p_seq, h_seq = m.forward_sequential(x_, b_, c_, a_)
    p_chunk, h_chunk = m.forward_chunk(x_, b_, c_, a_)
    assert torch.allclose(p_seq, p_chunk, atol=1e-5), "SSD: sequential ≠ chunk"
    assert torch.allclose(h_seq, h_chunk, atol=1e-5), "SSD: 末状态不一致"

    h = m._new_state(x_)
    outs = []
    for t in range(x_.shape[2]):
        y, h = m.forward_recurrent(x_[:, :, t:t + 1], b_[:, :, t:t + 1], c_[:, :, t:t + 1],
                                   a_[:, :, t:t + 1], h)
        outs.append(y[:, :, 0])
    p_rec = torch.stack(outs, dim=2)
    assert torch.allclose(p_seq, p_rec, atol=1e-5), "SSD: sequential ≠ recurrent"
    assert torch.allclose(h_seq, h, atol=1e-5)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["gdn_hybrid", "mamba_hybrid"])
def test_hybrid_presets_forward_backward(preset):
    m = build_model(preset)
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    logits = m(ids)
    assert logits.shape == (2, 16, m.cfg.vocab_size)
    logits.float().mean().backward()
    assert m.tok_embeddings.weight.grad is not None


@pytest.mark.parametrize("preset", ["gdn_hybrid", "mamba_hybrid"])
def test_hybrid_decode_does_not_grow_with_context(preset):
    """线性层的状态与上下文长度无关——这是它存在的全部理由，必须量化验证。"""
    m = build_model(preset).eval()
    m = m.float()
    h, d = m.cache_spec()

    def run(ctx: int, new: int):
        torch.manual_seed(0)
        cache = DenseKVCache(m.cfg.n_layers, 1, ctx + new + 8, h, d, dtype=torch.float32)
        x = torch.randint(0, m.cfg.vocab_size, (1, ctx))
        with torch.no_grad():
            m(x, cache=cache)
            cache.advance(ctx)
            out = []
            cur = x[:, -1:]
            for _ in range(new):
                lg = m(cur, cache=cache)
                cache.advance(1)
                cur = lg[:, -1].argmax(-1, keepdim=True)
                out.append(int(cur.item()))
        return out, cache

    _, c1 = run(8, 4)
    _, c2 = run(64, 4)
    # 递归状态的大小不随上下文变化（attention 层的 KV 是另一回事，只看 states）
    n1 = sum(t.numel() for t in c1.states.values())
    n2 = sum(t.numel() for t in c2.states.values())
    assert n1 == n2 > 0, f"递归状态随上下文增长了: {n1} -> {n2}"


def test_recurrent_state_bytes_reported():
    m = build_model("gdn_hybrid")
    assert m.recurrent_state_bytes() > 0
    assert isinstance(m.mixer_histogram, dict)
