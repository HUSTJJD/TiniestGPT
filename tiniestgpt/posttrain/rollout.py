"""Rollout 引擎：GRPO 闭环里"采样"那一半。

``posttrain/grpo.py`` 只提供了**损失函数**，但没有闭环——
没有采样就没有奖励，没有奖励就没有优势，loss 再正确也只是数学式。

本模块补上缺失的一半：对每个 prompt 采样 G 条回答，
并把采样时的 token log-prob 一并记下来（GRPO 的重要性采样比需要它）::

    prompt ──► [G 条采样] ──► rewards ──► 组内标准化 ──► grpo_loss

工程要点（也是生产系统 RL 后训练的主要复杂度）：

* 采样必须**批量**跑，否则 G×P 次单条生成慢到不可用；
* 采样时的 logp 必须存下来当 ``old_logps``（PPO 的重要性采样比）；
* 每一步的 mask 要正确处理 EOS——**EOS 之后的 token 不能进 loss**；
* 训练引擎与推理引擎的权重同步（生产上用 sleep mode / 权重广播），
  这里两者是同一份对象，所以天然同步。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = ["RolloutConfig", "Rollout", "RolloutEngine"]


@dataclass
class RolloutConfig:
    group_size: int = 8             # 每个 prompt 采样几条（GRPO 的 G）
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    device: str = "auto"
    include_eos: bool = True        # EOS 本身是否计入 loss


@dataclass
class Rollout:
    """一个 prompt 的一整组采样。"""

    prompt: str
    prompt_ids: List[int]
    samples: List[List[int]]               # G 条 completion（不含 prompt）
    texts: List[str]
    old_logps: torch.Tensor                # [G, T]
    mask: torch.Tensor                     # [G, T]
    rewards: Optional[torch.Tensor] = None  # [G]
    advantages: Optional[torch.Tensor] = None

    @property
    def group_size(self) -> int:
        return len(self.samples)

    @property
    def mean_reward(self) -> float:
        return float(self.rewards.mean().item()) if self.rewards is not None else 0.0


class RolloutEngine:
    """批量采样器：一次前向服务整个 group。"""

    def __init__(self, model, tokenizer, cfg: Optional[RolloutConfig] = None) -> None:
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg or RolloutConfig()
        dev = self.cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model.to(self.device).eval()
        self.stats: Dict[str, float] = {"rollouts": 0, "samples": 0, "ms": 0.0}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_group(self, prompt: str) -> Rollout:
        cfg = self.cfg
        G = cfg.group_size
        ids = self.tok.encode(prompt)
        if not ids:
            ids = [0]
        eos = getattr(self.tok, "eos_id", None)
        pad = getattr(self.tok, "pad_id", 0)

        x = torch.tensor([ids] * G, device=self.device)
        cache = self._new_cache(G, len(ids) + cfg.max_new_tokens + 8)
        logits = self.model(x, cache=cache)
        if cache is not None:
            cache.advance(len(ids))

        seqs: List[List[int]] = [[] for _ in range(G)]
        logps: List[torch.Tensor] = []
        alive = [True] * G
        for _ in range(cfg.max_new_tokens):
            nxt, lp = self._sample(logits[:, -1])
            logps.append(lp)
            for g in range(G):
                if alive[g]:
                    t = int(nxt[g].item())
                    seqs[g].append(t)
                    if eos is not None and t == eos:
                        alive[g] = False
            if not any(alive):
                break
            x = nxt.unsqueeze(1)
            logits = self.model(x, cache=cache)
            if cache is not None:
                cache.advance(1)

        T = max(len(s) for s in seqs)
        lp_t = torch.zeros(G, T, device=self.device)
        mask = torch.zeros(G, T, device=self.device)
        for g, s in enumerate(seqs):
            n = len(s)
            if n == 0:
                continue
            lp_t[g, :n] = torch.stack([p[g] for p in logps[:n]])
            keep = n if cfg.include_eos else (n - 1 if (eos is not None and s[-1] == eos) else n)
            mask[g, :max(keep, 0)] = 1.0

        texts = [self.tok.decode(s) for s in seqs]
        self.stats["rollouts"] += 1
        self.stats["samples"] += G
        return Rollout(prompt=prompt, prompt_ids=ids, samples=seqs, texts=texts,
                       old_logps=lp_t, mask=mask)

    def sample(self, prompts: Sequence[str]) -> List[Rollout]:
        t0 = time.time()
        out = [self.sample_group(p) for p in prompts]
        self.stats["ms"] += (time.time() - t0) * 1000
        return out

    # ------------------------------------------------------------------ #
    def _new_cache(self, batch: int, max_len: int):
        try:
            from ..model.kv_cache import DenseKVCache

            h, d = self.model.cache_spec()
            return DenseKVCache(self.model.cfg.n_layers, batch, max_len, h, d,
                                dtype=torch.float32, device=self.device)
        except Exception:
            return None

    def _sample(self, logits: torch.Tensor):
        """返回 ``(next_ids [G], logp [G])``。"""
        cfg = self.cfg
        x = logits.float()
        if cfg.temperature <= 1e-6:
            nxt = torch.argmax(x, dim=-1)
            return nxt, torch.zeros_like(nxt, dtype=torch.float32)
        x = x / cfg.temperature
        if cfg.top_k and cfg.top_k < x.shape[-1]:
            v, _ = torch.topk(x, cfg.top_k, dim=-1)
            x = torch.where(x < v[..., -1:], torch.full_like(x, -float("inf")), x)
        if 0.0 < cfg.top_p < 1.0:
            srt, idx = torch.sort(x, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1)
            rem = cum - torch.softmax(srt, dim=-1) > cfg.top_p
            srt = srt.masked_fill(rem, -float("inf"))
            x = torch.zeros_like(x).scatter(-1, idx, srt)
        probs = torch.softmax(x, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        lp = torch.log(probs.gather(-1, nxt.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12))
        return nxt, lp

    def report(self) -> str:
        s = self.stats
        return (f"rollout: {int(s['rollouts'])} prompts / {int(s['samples'])} samples "
                f"/ {s['ms']:.0f} ms / G={self.cfg.group_size}")
