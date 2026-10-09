"""MTP 自草稿推测解码（self-speculative decoding）。

与 :mod:`.speculative` 的区别：**不需要独立的 draft 模型**。

MTP head k 从位置 t 的隐藏状态直接预测 ``x_{t+2+k}``，
所以一次前向就能白拿 n 个候选 token::

    一次前向 → 主 logits(x_{t+1}) + MTP 草稿 [d_2, d_3, ..., d_{n+1}]
    再一次前向（把这 n+1 个候选喂进去）→ 批量验证 → 接受-拒绝

优势：没有第二份权重、没有第二个 KV Cache、无需维护两模型同步。
代价：草稿质量受限于 MTP 头本身，接受率通常低于同规模的外挂 draft 模型。

**不保证加速**：与所有投机解码一样，输出分布与逐 token 采样**严格一致**，
但 wall-clock 收益取决于接受率、batch 大小与验证阶段的吞吐。
``benchmarks/inference_ablation.py`` 的 L6 档会把这点量化出来。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..model.kv_cache import DenseKVCache
from .sampler import SamplingParams, Sampler, min_p_filter, top_k_top_p_filter

__all__ = ["MTPSpecStats", "MTPSpecDecoder"]


@dataclass
class MTPSpecStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    target_forwards: int = 0
    elapsed_ms: float = 0.0
    bonus_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.draft_tokens, 1)

    @property
    def mean_accept_length(self) -> float:
        """平均接受长度 τ：每轮主模型前向换来的 token 数（>1 才有收益）。"""
        return (self.accepted_tokens + self.bonus_tokens) / max(self.rounds, 1)

    def report(self) -> str:
        return (f"mtp-spec: rounds={self.rounds} draft={self.draft_tokens} "
                f"accepted={self.accepted_tokens} α={self.acceptance_rate:.2%} "
                f"τ={self.mean_accept_length:.2f} tok/forward "
                f"| {self.elapsed_ms:.0f}ms")


class MTPSpecDecoder:
    """用模型自带的 MTP 头做草稿的推测解码。"""

    def __init__(self, model, tokenizer=None, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        if getattr(model, "mtp", None) is None:
            raise ValueError("模型没有启用 MTP（ModelConfig.mtp_enabled=True 才能自草稿）")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.sampler = Sampler(self.device)
        self.generator: Optional[torch.Generator] = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        x = logits.float()
        if params.temperature <= 1e-6:
            return F.one_hot(torch.argmax(x), x.shape[-1]).float()
        x = x / params.temperature
        x = min_p_filter(x, params.min_p)
        x = top_k_top_p_filter(x, params.top_k, params.top_p)
        return F.softmax(x, dim=-1)

    def _new_cache(self, max_len: int) -> DenseKVCache:
        h, d = self.model.cache_spec()
        return DenseKVCache(self.model.cfg.n_layers, 1, max_len, h, d,
                            dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt_ids: List[int], params: Optional[SamplingParams] = None,
                 num_draft: Optional[int] = None, max_new_tokens: int = 64,
                 seed: Optional[int] = None) -> Dict:
        params = params or SamplingParams(max_tokens=max_new_tokens)
        n_draft = num_draft or self.model.cfg.mtp_n_predict
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

        t0 = time.time()
        max_len = len(prompt_ids) + max_new_tokens + n_draft + 8
        cache = self._new_cache(max_len)
        stats = MTPSpecStats()

        x = torch.tensor([prompt_ids], device=self.device)
        logits = self.model(x, cache=cache, return_mtp=True)
        cache.advance(len(prompt_ids))
        stats.target_forwards += 1

        first = self._sample(self._probs(logits[0, -1], params))
        outputs = [first]
        self._write([first], cache, len(prompt_ids))
        cached = len(prompt_ids) + 1

        while len(outputs) < max_new_tokens:
            # ---------- 1) 一次前向同时拿到主分布 + n 个 MTP 草稿 ----------
            xi = torch.tensor([[outputs[-1]]], device=self.device)
            pos = torch.tensor([[cached - 1]], device=self.device)
            cache.write_pos = cached - 1
            lg = self.model(xi, positions=pos, cache=cache, return_mtp=True)
            mtp = self.model.last_mtp_logits or []
            cache.advance(1)
            stats.target_forwards += 1

            p_main = self._probs(lg[0, -1], params)
            # MTP[k] 在位置 t 预测 x_{t+2+k}，即这里的第 k+2 个未来 token
            drafts, q_probs = [], []
            for k, mlg in enumerate(mtp[:n_draft]):
                q = self._probs(mlg[0, -1], params)
                drafts.append(self._sample(q))
                q_probs.append(q)
            if not drafts:
                outputs.append(self._sample(p_main))
                stats.bonus_tokens += 1
                self._write([outputs[-1]], cache, cached - 1)
                cached += 1
                continue
            stats.draft_tokens += len(drafts)
            stats.rounds += 1

            # ---------- 2) 一次前向批量验证 ----------
            seq = torch.tensor([[outputs[-1]] + drafts], device=self.device)
            pos = torch.arange(cached - 1, cached + len(drafts), device=self.device).unsqueeze(0)
            cache.write_pos = cached - 1
            t_logits = self.model(seq, positions=pos, cache=cache)
            cache.advance(len(drafts) + 1)
            stats.target_forwards += 1

            # ---------- 3) 接受-拒绝（保证分布严格等于 target） ----------
            accepted, rejected = 0, False
            for i, d_tok in enumerate(drafts):
                p = self._probs(t_logits[0, i], params)
                q = q_probs[i]
                pd, qd = p[d_tok].item(), q[d_tok].item()
                if qd <= 0 or self._uniform() < min(1.0, pd / max(qd, 1e-9)):
                    outputs.append(d_tok)
                    accepted += 1
                else:
                    resid = torch.clamp(p - q, min=0.0)
                    resid = resid / resid.sum().clamp(min=1e-9)
                    outputs.append(self._sample(resid))
                    rejected = True
                    break
            if not rejected:
                # 草稿全中 → 白拿一个 bonus token
                outputs.append(self._sample(self._probs(t_logits[0, -1], params)))
                stats.bonus_tokens += 1
            stats.accepted_tokens += accepted

            # ---------- 4) 把最后一个 token 写回缓存 ----------
            pos_w = len(prompt_ids) + len(outputs) - 1
            cache.write_pos = pos_w
            self._write([outputs[-1]], cache, pos_w)
            cached = pos_w + 1

            if outputs[-1] in params.stop_token_ids and not params.ignore_eos:
                break

        stats.elapsed_ms = (time.time() - t0) * 1000
        return {"output_ids": outputs,
                "text": self.tokenizer.decode(outputs) if self.tokenizer else "",
                "stats": stats}

    # ------------------------------------------------------------------ #
    def _write(self, tokens: List[int], cache: DenseKVCache, pos: int) -> None:
        x = torch.tensor([tokens], device=self.device)
        p = torch.arange(pos, pos + len(tokens), device=self.device).unsqueeze(0)
        self.model(x, positions=p, cache=cache)
        cache.advance(len(tokens))

    def _uniform(self) -> float:
        if self.generator is not None:
            return float(torch.rand(1, generator=self.generator).item())
        return float(torch.rand(1).item())

    def _sample(self, probs: torch.Tensor) -> int:
        if self.generator is not None and probs.device.type == "cpu":
            return int(torch.multinomial(probs, 1, generator=self.generator).item())
        return int(torch.multinomial(probs, 1).item())
