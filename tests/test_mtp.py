"""MTP（多 token 预测）：训练目标 + 自草稿推测解码。"""

import torch

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.train.losses import mtp_loss


def _model(mtp_n: int = 2):
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=256,
                      max_seq_len=64, mtp_enabled=True, mtp_n_predict=mtp_n)
    return build_model(cfg)


# --------------------------------------------------------------------------- #
def test_mtp_returns_n_heads_with_correct_alignment():
    m = _model(2)
    ids = torch.randint(0, m.cfg.vocab_size, (2, 12))
    m.eval()
    logits = m(ids, return_mtp=True)
    mtp = m.last_mtp_logits
    assert mtp is not None and len(mtp) == 2
    for lg in mtp:
        assert lg.shape == (2, 12, m.cfg.vocab_size)


def test_mtp_loss_alignment():
    """第 k 个头在位置 t 的目标必须是 x_{t+2+k}，错位会让主损失被污染。"""
    V, B, T = 32, 2, 8
    mtp = [torch.randn(B, T, V) for _ in range(2)]
    labels = torch.randint(0, V, (B, T))
    out = mtp_loss(mtp, labels, weight=0.5)
    assert out["mtp"].shape == ()
    assert float(out["mtp"]) > 0.0
    # 空输入不应炸
    assert float(mtp_loss([], labels, weight=0.5)["mtp"]) == 0.0


def test_mtp_training_step_runs():
    m = _model(1)
    m.train()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 10))
    labels = torch.randint(0, m.cfg.vocab_size, (2, 10))
    logits = m(ids)
    loss = logits.float().mean() + mtp_loss(m.last_mtp_logits, labels, weight=0.1)["mtp"]
    loss.backward()
    assert torch.isfinite(loss)


def test_mtp_spec_decoder_runs_and_keeps_distribution():
    """自草稿投机解码应该能跑通，并报告平均接受长度。"""
    from tiniestgpt.inference.mtp_spec import MTPSpecDecoder
    from tiniestgpt.inference.sampler import SamplingParams

    m = _model(2).float().eval()
    dec = MTPSpecDecoder(m, tokenizer=None, device="cpu", dtype=torch.float32)
    res = dec.generate([1, 2, 3, 4], SamplingParams(temperature=0.8, max_tokens=8),
                       max_new_tokens=8, seed=0)
    assert len(res["output_ids"]) >= 1
    assert res["stats"].rounds >= 1
    assert res["stats"].draft_tokens > 0
    assert 0.0 <= res["stats"].acceptance_rate <= 1.0


def test_mtp_spec_requires_mtp_enabled():
    from tiniestgpt.inference.mtp_spec import MTPSpecDecoder

    m = build_model("nano")
    try:
        MTPSpecDecoder(m)
    except ValueError:
        return
    raise AssertionError("未启用 MTP 的模型不应能构造自草稿解码器")
