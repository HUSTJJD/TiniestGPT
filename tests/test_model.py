"""模型测试：形状、KV Cache 一致性、GQA/MLA/MoE/线性注意力、RoPE。"""

import torch

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.model.kv_cache import DenseKVCache, PagedKVCache
from tiniestgpt.model.rope import RotaryEmbedding, apply_rotary_emb


def _cfg(**kw):
    return ModelConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       hidden_dim=128, max_seq_len=128, **kw)


def test_forward_shape():
    m = build_model(_cfg())
    x = torch.randint(0, 512, (2, 16))
    logits = m(x)
    assert logits.shape == (2, 16, 512)


def test_gqa_reduces_kv_heads():
    cfg = _cfg(attn_type="gqa")
    m = build_model(cfg)
    assert m.cfg.n_kv_heads == 2 and m.cfg.n_groups == 2
    assert m.cache_spec() == (2, cfg.head_dim)


def test_dense_kv_cache_matches_full_forward():
    """核心一致性：prefill + 逐步 decode 的结果必须等于一次性全序列前向。"""
    torch.manual_seed(0)
    cfg = _cfg(attn_type="gqa", attn_window=-1)
    m = build_model(cfg).eval()
    ids = torch.randint(0, 512, (1, 12))

    with torch.no_grad():
        full = m(ids)

        cache = DenseKVCache(cfg.n_layers, 1, 32, 2, cfg.head_dim, dtype=torch.float32)
        out = m(ids[:, :8], cache=cache)
        cache.advance(8)
        outs = [out]
        for i in range(8, 12):
            o = m(ids[:, i:i + 1], positions=torch.tensor([[i]]), cache=cache)
            cache.advance(1)
            outs.append(o)
    cat = torch.cat(outs, dim=1)
    assert torch.allclose(full, cat, atol=1e-4), (full - cat).abs().max()


def test_paged_kv_cache_matches_dense():
    """分页缓存 + PagedAttention 的结果必须与稠密缓存一致。"""
    torch.manual_seed(0)
    cfg = _cfg(attn_type="gqa")
    m = build_model(cfg).eval()
    ids = torch.randint(0, 512, (1, 10))

    with torch.no_grad():
        dense = DenseKVCache(cfg.n_layers, 1, 32, 2, cfg.head_dim, dtype=torch.float32)
        ref = m(ids[:, :10], cache=dense)
        dense.advance(10)
        d_next = m(ids[:, -1:], positions=torch.tensor([[10]]), cache=dense)

        paged = PagedKVCache(cfg.n_layers, num_blocks=8, block_size=4, n_kv_heads=2,
                             head_dim=cfg.head_dim, dtype=torch.float32)
        blocks = paged.allocator.allocate(4)              # 4 blocks * 4 = 16 slots
        bt = torch.tensor([blocks], dtype=torch.long)
        slots = torch.tensor([[blocks[i // 4] * 4 + i % 4 for i in range(10)]], dtype=torch.long)
        paged.set_batch(block_table=bt, seq_lens=torch.tensor([10]), slot_ids=slots)
        p_ref = m(ids[:, :10], cache=paged)

        # decode：第 11 个 token
        slot11 = torch.tensor([[blocks[10 // 4] * 4 + 10 % 4]], dtype=torch.long)
        paged.set_batch(block_table=bt, seq_lens=torch.tensor([11]), slot_ids=slot11)
        p_next = m(ids[:, -1:], positions=torch.tensor([[10]]), cache=paged)

    assert torch.allclose(ref, p_ref, atol=1e-4), (ref - p_ref).abs().max()
    assert torch.allclose(d_next, p_next, atol=1e-4), (d_next - p_next).abs().max()


def test_sliding_window_and_sink_mask():
    """窗口外不可见；sink token 永远可见。"""
    torch.manual_seed(0)
    cfg = _cfg(attn_window=4, attn_sinks=2)
    m = build_model(cfg).eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        out_win = m(ids)
    cfg2 = _cfg(attn_window=-1)
    m2 = build_model(cfg2).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        out_full = m2(ids)
    # 窗口注意力与全注意力结果不同（说明窗口确实生效了）
    assert not torch.allclose(out_win, out_full, atol=1e-4)


def test_mla_forward_and_cache_spec():
    cfg = _cfg(attn_type="mla", mla_latent_dim=16, mla_rope_dim=8)
    m = build_model(cfg)
    h, d = m.cache_spec()
    # MLA 每 token 只缓存一条 latent + 解耦的 rope 向量
    assert h == 1 and d == 16 + 8, (h, d)
    ids = torch.randint(0, 512, (2, 8))
    out = m(ids)
    assert out.shape == (2, 8, 512)
    # 与普通 GQA 相比，MLA 的 KV 缓存应显著更小
    gqa = build_model(_cfg(attn_type="gqa"))
    assert m.kv_cache_bytes_per_token(torch.float16) < gqa.kv_cache_bytes_per_token(torch.float16)


def test_moe_forward_and_balance():
    cfg = _cfg(moe_enabled=True, n_experts=4, n_experts_per_tok=2, n_shared_experts=1)
    m = build_model(cfg).eval()
    ids = torch.randint(0, 512, (2, 8))
    out = m(ids)
    assert out.shape == (2, 8, 512)
    assert m.last_aux_loss is not None
    stats = m.layers[0].moe_stats
    assert stats.load.shape == (4,)
    assert int(stats.load.sum()) == 2 * 8 * cfg.n_experts_per_tok   # B*T*topk


def test_linear_attention_parallel_equals_recurrent():
    """线性注意力的并行形态与递推形态必须数学等价（RetNet 的核心性质）。"""
    from tiniestgpt.model.linear_attention import LinearAttention

    torch.manual_seed(0)
    cfg = _cfg()
    la = LinearAttention(cfg, 0).eval()
    x = torch.randn(1, 6, cfg.dim)
    with torch.no_grad():
        parallel = la(x)
        # 递推形态：逐 token 喂进去，状态存在 cache.states 里
        states: dict = {}
        cache = _DummyCache(states)
        rec = []
        for t in range(6):
            rec.append(la(x[:, t:t + 1], cache=cache))
        rec = torch.cat(rec, dim=1)
    assert torch.allclose(parallel, rec, atol=1e-4), (parallel - rec).abs().max()


class _DummyCache:
    def __init__(self, states):
        self.states = states


def test_rope_relative_property():
    """RoPE 的关键性质：q·k 只依赖**相对**位置（绝对位置平移不影响点积）。"""
    rope = RotaryEmbedding(head_dim=16, rope_type="rope", max_seq_len=64)
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 1, 16)
    # 同一对向量，分别放在 (0, 2) 与 (5, 7) —— 相对距离都是 -2
    q_at0 = rope(q, q, torch.tensor([[0]]))[0]
    k_at2 = rope(k, k, torch.tensor([[2]]))[0]
    q_at5 = rope(q, q, torch.tensor([[5]]))[0]
    k_at7 = rope(k, k, torch.tensor([[7]]))[0]
    d1 = (q_at0 * k_at2).sum()
    d2 = (q_at5 * k_at7).sum()
    assert torch.allclose(d1, d2, atol=1e-4), (d1, d2)


def test_rope_partial_and_yarn():
    rope = RotaryEmbedding(head_dim=16, rope_type="yarn", scaling=4.0,
                           original_max_len=64, max_seq_len=256)
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    qo, ko = rope(q, k, torch.arange(4).unsqueeze(0))
    assert qo.shape == q.shape
    assert rope.attn_scale > 1.0


def test_apply_rotary_partial_dims():
    """部分旋转：只有前 D_rot 维被旋转，其余维原样保留。"""
    x = torch.randn(1, 4, 2, 16).transpose(1, 2)      # [B, H, T, D]
    cos = torch.randn(1, 1, 4, 8)                     # [B, 1, T, D_rot]
    sin = torch.randn(1, 1, 4, 8)
    y = apply_rotary_emb(x, cos, sin)
    assert y.shape == x.shape
    assert torch.equal(y[..., 8:], x[..., 8:])        # 未参与旋转的维度不变


def test_active_params_with_moe():
    cfg = _cfg(moe_enabled=True, n_experts=4, n_experts_per_tok=2)
    m = build_model(cfg)
    assert m.active_params < m.num_params()["total"]
