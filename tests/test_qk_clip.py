"""MuonClip / QK-Clip：约束前向 logit，而不是约束梯度。"""

import torch

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.train.qk_clip import QKClipGuard, apply_qk_clip, max_attn_logits


def _model():
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                      max_seq_len=64, qk_norm=False)
    return build_model(cfg).float()


def _max_logit(m, ids):
    """独立算一遍真实的 attention logit（用于验证 clip 确实只做了标量缩放）。

    注意 GQA：K 的头数少于 Q，必须先 repeat 到对齐，否则逐 head 的 logit 是错的。
    """
    blk = m.layers[0]
    with torch.no_grad():
        h = blk.norm1(m.tok_embeddings(ids))
        q = blk.mixer.q_proj(h)
        k = blk.mixer.k_proj(h)
    H, D = m.cfg.n_heads, m.cfg.head_dim
    qh = q.view(*q.shape[:-1], H, D)
    hk = k.shape[-1] // D
    kh = k.view(*k.shape[:-1], hk, D).repeat_interleave(H // hk, dim=-2)
    return torch.einsum("bthd,bshd->bhts", qh, kh) / (D ** 0.5)


# --------------------------------------------------------------------------- #
def test_max_attn_logits_shape():
    m = _model()
    ids = torch.randint(0, 64, (2, 8))
    out = max_attn_logits(m, ids)
    assert out, "没抓到任何 q_proj/k_proj 模块"
    for name, s in out.items():
        assert s.shape == (m.cfg.n_heads,), f"{name}: {s.shape}"


def test_qk_clip_shrinks_logits_to_threshold():
    m = _model()
    ids = torch.randint(0, 64, (2, 8))
    before = _max_logit(m, ids).abs().max().item()

    s_max = max_attn_logits(m, ids)
    tau = max(before / 4.0, 1e-3)          # 阈值远低于当前最大值 → 必然触发
    stats = apply_qk_clip(m, s_max, tau=tau, alpha=0.5)
    assert stats and min(stats.values()) < 1.0, "clip 没有触发"

    after = _max_logit(m, ids).abs().max().item()
    assert after < before, f"logit 没有下降: {before} -> {after}"


def test_qk_clip_preserves_relative_ordering():
    """缩放必须**逐 head** 且只改绝对尺度——不能改变 head 内的相对分布。"""
    m = _model()
    ids = torch.randint(0, 64, (2, 8))
    s0 = _max_logit(m, ids)[0, 0]              # [T, T]
    s_max = max_attn_logits(m, ids)
    apply_qk_clip(m, s_max, tau=1e-3, alpha=0.5)
    s1 = _max_logit(m, ids)[0, 0]
    # 逐元素比例应该是常数（Q/K 各乘 gamma^0.5 → 点积整体乘 gamma）
    ratio = (s1.abs().clamp(min=1e-8) / s0.abs().clamp(min=1e-8))
    assert torch.allclose(ratio, ratio.mean(), rtol=1e-3), "缩放不是纯标量缩放"


def test_guard_observes_and_reports():
    m = _model()
    guard = QKClipGuard(m, tau=1e-3, alpha=0.5, every=1).install()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        m(ids)
    assert guard.running, "hook 没有记录到 logit"
    rec = guard.maybe_clip(1)
    assert rec is not None and rec["n_modules"] > 0
    assert "qk-clip" in guard.report()
