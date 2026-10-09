"""GRPO 训练器：把"采样 → 打分 → 更新"真正闭环起来。

``posttrain/grpo.py`` 给出的是**损失函数**；这个文件给出的是**训练循环**。
两者缺一不可——这也是很多"实现了 GRPO"的教学代码真正缺的那一半。

一个 step 的完整数据流::

    prompts ──► rollout(G 条/题) ──► rule/RM 打分 ──► 组内标准化优势
        └──────────────────────────────────────────────► grpo_loss ──► 更新

三处容易踩的坑（都在代码里显式处理了）：

1. **rollout 后策略已经变了**。``old_logps`` 必须是**采样时**的 logp，
   否则 PPO 的重要性采样比失去意义；
2. **EOS 之后的 token 不能进 loss**，否则模型会学到"生成垃圾也能拿分"；
3. **KL 不是必需的**。DeepSeek-R1 / DAPO 都去掉了 KL（beta=0），
   换来更快的探索与更大的更新幅度；需要保守时再打开。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .grpo import GRPOConfig, compute_group_advantages, grpo_loss
from .reward import RuleReward
from .rollout import Rollout, RolloutConfig, RolloutEngine

log = logging.getLogger(__name__)

__all__ = ["GRPOTrainConfig", "GRPOTrainer"]


@dataclass
class GRPOTrainConfig:
    group_size: int = 8
    prompts_per_step: int = 4          # 每步用多少个 prompt（×G = 该步的样本数）
    inner_epochs: int = 1              # 每批 rollout 复用几次（>1 时 old_logps 才真正起作用）
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 100
    log_every: int = 10
    eval_every: int = 0
    reward_kind: str = "exact_int"     # 见 RuleReward.KINDS
    beta: float = 0.0                  # KL 系数；0 = 不用 KL（DAPO / R1 的做法）
    eps_clip: float = 0.2
    clip_higher: float = 0.28
    device: str = "auto"
    seed: int = 0

    def rollout_cfg(self, **kw) -> RolloutConfig:
        return RolloutConfig(group_size=self.group_size, device=self.device, **kw)

    def grpo_cfg(self) -> GRPOConfig:
        return GRPOConfig(group_size=self.group_size, beta=self.beta,
                          eps_clip=self.eps_clip, clip_higher=self.clip_higher)


class GRPOTrainer:
    """最小可用的 GRPO 闭环。

    :param prompts: 可调用对象或列表，提供 prompt 与参考答案
                    ``(prompt, answer)`` 二元组
    """

    def __init__(self, model, tokenizer, prompts: Sequence[Any],
                 cfg: Optional[GRPOTrainConfig] = None,
                 reward: Optional[Callable[[str, str, Any], float]] = None,
                 ref_model=None) -> None:
        self.cfg = cfg or GRPOTrainConfig()
        self.model = model
        self.tok = tokenizer
        self.prompts = list(prompts)
        self.reward = reward or RuleReward(kind=self.cfg.reward_kind)
        self.ref_model = ref_model

        dev = self.cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device).eval()

        self.rollout = RolloutEngine(self.model, self.tok,
                                     self.cfg.rollout_cfg())
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay)
        self.history: List[Dict[str, float]] = []
        self.step = 0

    # ------------------------------------------------------------------ #
    def _batch_prompts(self) -> List[Any]:
        n = self.cfg.prompts_per_step
        start = (self.step * n) % max(len(self.prompts), 1)
        return [self.prompts[(start + i) % len(self.prompts)] for i in range(n)]

    # ------------------------------------------------------------------ #
    def _sequence_logps(self, roll: Rollout) -> torch.Tensor:
        """把采样序列重新前向一遍，得到**当前策略**的 token logp ``[G,T]``。"""
        G = roll.group_size
        T = roll.old_logps.shape[1]
        seqs = []
        for g in range(G):
            seqs.append(torch.tensor(roll.prompt_ids + roll.samples[g][:T], device=self.device))
        maxlen = max(len(s) for s in seqs)
        x = torch.full((G, maxlen), getattr(self.tok, "pad_id", 0), dtype=torch.long,
                       device=self.device)
        for g, s in enumerate(seqs):
            x[g, : len(s)] = s
        logits = self.model(x)
        lp = torch.log_softmax(logits.float(), dim=-1)
        out = torch.zeros(G, T, device=self.device)
        for g, s in enumerate(seqs):
            p = len(roll.prompt_ids)
            tgt = s[p:p + T]
            out[g, : len(tgt)] = lp[g, p - 1:p - 1 + len(tgt)].gather(
                -1, tgt.unsqueeze(-1)).squeeze(-1)
        return out

    # ------------------------------------------------------------------ #
    def train_step(self) -> Dict[str, float]:
        cfg = self.cfg
        gcfg = cfg.grpo_cfg()
        batch = self._batch_prompts()
        t0 = time.time()

        # ---------- 1) 采样 ----------
        rolls: List[Rollout] = self.rollout.sample([p for p, _ in batch])

        # ---------- 2) 打分 + 组内优势 ----------
        for roll, (_p, ans) in zip(rolls, batch):
            r = torch.tensor([self.reward(roll.prompt, t, ans) for t in roll.texts],
                             dtype=torch.float32, device=self.device)
            roll.rewards = r
            roll.advantages = compute_group_advantages(r, roll.group_size)

        # ---------- 3) 更新（可复用 inner_epochs 次） ----------
        tot_loss, tot_kl, tot_clip, n_batches = 0.0, 0.0, 0.0, 0
        for _ in range(cfg.inner_epochs):
            for roll in rolls:
                logps = self._sequence_logps(roll)
                ref = None
                if self.ref_model is not None and gcfg.beta > 0:
                    with torch.no_grad():
                        ref = self._ref_logps(roll)
                out = grpo_loss(logps, roll.old_logps.detach(), roll.advantages,
                                roll.mask, ref_logps=ref, cfg=gcfg)
                (out["loss"] / max(len(rolls), 1)).backward()
                tot_loss += float(out["loss"].item())
                tot_kl += float(out["kl"].item())
                tot_clip += float(out["clip_frac"].item())
                n_batches += 1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        self.step += 1
        rewards = torch.stack([r.rewards for r in rolls])
        stats = {
            "step": self.step,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "reward_max": float(rewards.max().item()),
            "loss": tot_loss / max(n_batches, 1),
            "kl": tot_kl / max(n_batches, 1),
            "clip_frac": tot_clip / max(n_batches, 1),
            "mean_len": float(torch.stack([r.mask.sum(-1) for r in rolls]).mean().item()),
            "ms": (time.time() - t0) * 1000,
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _ref_logps(self, roll: Rollout) -> torch.Tensor:
        was = self.model
        self.model = self.ref_model
        try:
            return self._sequence_logps(roll)
        finally:
            self.model = was

    # ------------------------------------------------------------------ #
    def train(self, max_steps: Optional[int] = None) -> List[Dict[str, float]]:
        max_steps = max_steps or self.cfg.max_steps
        for _ in range(max_steps):
            s = self.train_step()
            if self.cfg.log_every and s["step"] % self.cfg.log_every == 0:
                log.info("grpo step %4d | reward %.3f±%.3f (max %.2f) | loss %.4f | "
                         "kl %.4f | clip %.2f | len %.1f | %.0fms",
                         s["step"], s["reward_mean"], s["reward_std"], s["reward_max"],
                         s["loss"], s["kl"], s["clip_frac"], s["mean_len"], s["ms"])
        return self.history

    def report(self) -> str:
        if not self.history:
            return "grpo: 尚未训练"
        first = self.history[: max(1, len(self.history) // 10)]
        last = self.history[-max(1, len(self.history) // 10):]
        r0 = sum(h["reward_mean"] for h in first) / len(first)
        r1 = sum(h["reward_mean"] for h in last) / len(last)
        return (f"GRPO: {len(self.history)} steps | reward {r0:.3f} → {r1:.3f} "
                f"| {self.rollout.report()}")
