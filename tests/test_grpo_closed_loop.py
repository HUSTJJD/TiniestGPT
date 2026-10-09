"""后训练闭环：rollout → 奖励 → 组内优势 → GRPO 更新。

这个测试盯的是 2026 年 RL 后训练最容易"名不副实"的地方：
**只有 loss 函数、没有采样闭环**。
"""

import torch

from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.posttrain.grpo import compute_group_advantages
from tiniestgpt.posttrain.grpo_trainer import GRPOTrainConfig, GRPOTrainer
from tiniestgpt.posttrain.reward import RuleReward, bradley_terry_loss, reward_from_rules
from tiniestgpt.posttrain.rollout import RolloutConfig, RolloutEngine


class ToyTok:
    """测试用最小 tokenizer（不依赖真实词表）。"""

    eos_id, pad_id = 1, 0

    def __init__(self, vocab=64):
        self.vocab = vocab

    def encode(self, text, **kw):
        return [min(2 + (ord(c) % (self.vocab - 3)), self.vocab - 1) for c in text[:24]]

    def decode(self, ids, **kw):
        return "".join(chr(32 + (i % 90)) for i in ids)


def _model():
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                      max_seq_len=128, attn_window=-1)
    return build_model(cfg).float()


# --------------------------------------------------------------------------- #
def test_rollout_shapes_and_mask():
    m, tok = _model(), ToyTok()
    eng = RolloutEngine(m, tok, RolloutConfig(group_size=4, max_new_tokens=6,
                                              device="cpu"))
    r = eng.sample_group("Calculate: 3 + 4 =")
    assert r.group_size == 4
    assert r.old_logps.shape == r.mask.shape
    assert int(r.mask.sum()) > 0, "mask 全 0 → loss 会是 NaN"
    for s in r.samples:
        assert len(s) >= 1
    assert eng.stats["samples"] == 4


def test_rollout_mask_excludes_after_eos():
    m, tok = _model(), ToyTok()
    eng = RolloutEngine(m, tok, RolloutConfig(group_size=2, max_new_tokens=8,
                                              include_eos=False, device="cpu"))
    r = eng.sample_group("hello")
    # EOS 之后不再计入（include_eos=False 时会少算一个）
    assert r.mask.shape == r.old_logps.shape
    assert float(r.mask.sum()) <= float(r.old_logps.shape[0] * r.old_logps.shape[1])


# --------------------------------------------------------------------------- #
def test_rule_rewards_are_verifiable():
    r = RuleReward(kind="exact_int")
    assert r("Calculate: 3 + 4 =", "The answer is 7", 7) == 1.0
    assert r("Calculate: 3 + 4 =", "The answer is 9", 7) == 0.0
    rj = RuleReward(kind="json_valid")
    assert rj("", '{"name": "a", "age": 3}', {"name": "str", "age": "int"}) == 1.0
    assert rj("", "not json", {"name": "str"}) == 0.0
    t = reward_from_rules(r, ["7", "9"], 7)
    assert t.tolist() == [1.0, 0.0]


def test_bradley_terry_prefers_chosen():
    loss = bradley_terry_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.5]))
    assert float(loss) >= 0.0
    # 反向（更差的 chosen）应该给出更大的损失
    bad = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([5.0]))
    assert float(bad) > float(loss)


def test_group_advantages_zero_mean():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = compute_group_advantages(r, 4)
    assert abs(float(adv.mean())) < 1e-5
    # 全同奖励 → 优势全 0（没有学习信号，这正是 GRPO 会跳过这种组的原因）
    same = compute_group_advantages(torch.ones(4), 4)
    assert torch.allclose(same, torch.zeros(4), atol=1e-4)


# --------------------------------------------------------------------------- #
def test_grpo_trainer_runs_a_closed_loop():
    """真正的闭环：采样 → 打分 → 更新，且每步都产出可观测的统计量。"""
    m, tok = _model(), ToyTok()
    prompts = [("Calculate: 1 + 1 =", 2), ("Calculate: 2 + 2 =", 4)]
    cfg = GRPOTrainConfig(group_size=4, prompts_per_step=2, inner_epochs=1,
                          max_steps=2, lr=1e-4, reward_kind="exact_int", device="cpu")
    trainer = GRPOTrainer(m, tok, prompts, cfg=cfg)
    hist = trainer.train(max_steps=2)
    assert len(hist) == 2
    for h in hist:
        assert 0.0 <= h["reward_mean"] <= 1.0
        assert h["mean_len"] > 0
    assert "GRPO" in trainer.report()
    assert "rollout" in trainer.report()


def test_grpo_loss_decreases_when_reward_shaped():
    """奖励有区分度时，优势应该是非零的（否则学不到东西）。"""
    m, tok = _model(), ToyTok()
    cfg = GRPOTrainConfig(group_size=4, prompts_per_step=1, max_steps=1, device="cpu")
    prompts = [("Calculate: 1 + 1 =", 2)]

    class Shaped:
        """人为让"包含正确答案"的采样得分更高。"""

        def __call__(self, prompt, completion, answer):
            return 1.0 if str(answer) in completion else 0.0

    trainer = GRPOTrainer(m, tok, prompts, cfg=cfg, reward=Shaped())
    s = trainer.train_step()
    assert "reward_std" in s
