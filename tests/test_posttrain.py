"""后训练测试：SFT 掩码、DPO 损失、GRPO 优势与裁剪、蒸馏损失。"""

import torch

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.posttrain.dpo import dpo_loss, sequence_logps
from tiniestgpt.posttrain.distill import hidden_state_loss, kd_loss
from tiniestgpt.posttrain.grpo import compute_group_advantages, grpo_loss, kl_penalty
from tiniestgpt.posttrain.sft import SFTConfig, SFTExample, build_sft_batch, sft_loss


class DummyTokenizer:
    pad_id, bos_id, eos_id = 0, 1, 2

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [(ord(c) % 50) + 3 for c in text[:20]] or [3]
        return ([1] if add_bos else []) + ids + ([2] if add_eos else [])

    def decode(self, ids, skip_special=True):
        return "".join(chr(int(i) + 40) for i in ids if int(i) > 2)


def test_sft_masks_prompt_tokens():
    tok = DummyTokenizer()
    cfg = SFTConfig(max_len=32, batch_size=2, packing=True)
    ex = [SFTExample(prompt="question here", response="answer here"),
          SFTExample(prompt="another question", response="another answer")]
    batch = build_sft_batch(ex, tok, cfg)
    labels = batch["labels"]
    # prompt 段被 mask（-100），response 段参与 loss
    n_prompt = len(tok.encode("question here", add_bos=True))
    assert (labels[0][:n_prompt - 1] == -100).all()
    assert (labels[0][n_prompt - 1:] != -100).any()
    # 不 mask prompt 时，所有 token 都参与 loss
    cfg_nomask = SFTConfig(max_len=32, batch_size=2, mask_prompt=False)
    b2 = build_sft_batch(ex, tok, cfg_nomask)
    assert (b2["labels"][0][:3] != -100).all()
    assert batch["attn_mask"] is not None


def test_sft_loss_ignore_index():
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[-100, 1, 2, 3], [4, -100, -100, 5]])
    loss = sft_loss(logits, labels)
    assert torch.isfinite(loss) and loss > 0


def test_dpo_loss_prefers_chosen():
    # 策略给 chosen 更高概率 → loss 应该较小
    pc = torch.tensor([-1.0, -1.0])
    pr = torch.tensor([-5.0, -5.0])
    rc = torch.tensor([-1.5, -1.5])
    rr = torch.tensor([-4.5, -4.5])
    out = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert out["accuracy"].item() == 1.0
    bad = dpo_loss(pr, pc, rr, rc, beta=0.1)
    assert bad["loss"] > out["loss"]


def test_sequence_logps_shape():
    m = build_model(ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                                n_kv_heads=2, hidden_dim=64, max_seq_len=32))
    ids = torch.randint(0, 64, (3, 8))
    labels = ids.clone()
    labels[:, :2] = -100
    lp = sequence_logps(m, ids, labels)
    assert lp.shape == (3,)


def test_grpo_advantages_zero_mean():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    adv = compute_group_advantages(rewards, group_size=3)
    assert torch.allclose(adv[:3].mean(), torch.tensor(0.0), atol=1e-5)
    # 组内方差越大，标准化后仍保持零均值
    assert adv.shape == rewards.shape


def test_grpo_loss_clipping():
    torch.manual_seed(0)
    N, T = 4, 6
    logps = torch.randn(N, T)
    old = logps.clone() + 0.01 * torch.randn(N, T)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(N, T)
    mask[:, -1] = 0
    out = grpo_loss(logps, old, adv, mask)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= out["clip_frac"] <= 1.0
    # ratio 与 1 很接近时，裁剪几乎不生效
    assert out["clip_frac"] < 0.1


def test_kl_penalty_nonneg():
    logp = torch.zeros(4)
    ref = torch.tensor([0.5, -0.5, 1.0, -1.0])
    k3 = kl_penalty(logp, ref, "k3")
    assert (k3 >= 0).all()
    k1 = kl_penalty(logp, ref, "k1")
    assert k1.shape == k3.shape


def test_kd_loss_temperature():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 8)
    t = torch.randn(2, 5, 8) * 0.5
    out = kd_loss(s, t, temperature=2.0, labels=torch.randint(0, 8, (2, 5)))
    assert torch.isfinite(out["loss"])
    assert "kd" in out and "ce" in out
    # 温度改变会显著改变 KD 项的量级（T² 缩放 + 分布平滑）
    a = kd_loss(s, t, temperature=1.0)["kd"].item()
    b = kd_loss(s, t, temperature=4.0)["kd"].item()
    assert abs(a - b) > 1e-6
    assert torch.isfinite(torch.tensor(a)) and torch.isfinite(torch.tensor(b))


def test_hidden_state_loss():
    from tiniestgpt.posttrain.distill import DistillProjector

    s = torch.randn(2, 4, 16)
    t = torch.randn(2, 4, 32)
    proj = DistillProjector(16, 32)
    loss = hidden_state_loss(s, t, proj)
    assert torch.isfinite(loss)
    loss_cos = hidden_state_loss(proj(s), t, None, kind="cosine")
    assert torch.isfinite(loss_cos)
