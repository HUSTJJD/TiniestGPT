"""投机解码（Speculative Decoding）。

动机：decode 是 **memory-bound**——每生成一个 token 都要把整个模型权重
（以及 KV）从 HBM 读一遍，但每次只做极少的有效计算。
投机解码用"**用小模型猜、大模型一次性验证**"把串行依赖变成并行：

1. draft 模型（小、快）自回归地猜 γ 个 token；
2. target 模型（大、准）**一次前向**算出这 γ 个位置的真实分布；
3. 逐个做**接受-拒绝采样**：
   ``accept with p = min(1, p_target(x) / p_draft(x))``；
   一旦拒绝，就从 ``(p_target - p_draft)_+`` 归一化后重采样；
4. 全部接受则再免费多得 1 个 token（bonus）。

数学保证：**输出分布与直接用 target 采样完全一致**（不是近似！），
这是它优于"beam 加速 / 早退"等方法的根本原因。
加速比取决于接受率 α：``加速 ≈ (1-α^{γ+1}) / ((1-α)·(c·γ + 1))``，
c 是 draft/target 的单步成本比。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..model.kv_cache import DenseKVCache
from .sampler import SamplingParams, Sampler, apply_penalties, top_k_top_p_filter, min_p_filter

__all__ = ["SpeculativeDecoder", "SpecDecodeStats"]


@dataclass
class SpecDecodeStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    elapsed_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.draft_tokens, 1)

    @property
    def tokens_per_target_forward(self) -> float:
        return (self.accepted_tokens + 0.0) / max(self.target_forwards, 1)

    def report(self) -> str:
        return (f"spec-decode: rounds={self.rounds} draft={self.draft_tokens} "
                f"accepted={self.accepted_tokens} α={self.acceptance_rate:.2%} "
                f"| {self.tokens_per_target_forward:.2f} tok/target-forward "
                f"| {self.elapsed_ms:.0f}ms")


class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, tokenizer=None,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.target = target_model.to(device).eval()
        self.draft = draft_model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.sampler = Sampler(self.device)
        self.generator: Optional[torch.Generator] = None

    # ------------------------------------------------------------------ #
    def _new_cache(self, model, batch: int, max_len: int, cache_dtype=None) -> DenseKVCache:
        h, d = model.cache_spec()
        return DenseKVCache(model.cfg.n_layers, batch, max_len, h, d,
                            dtype=cache_dtype or self.dtype, device=self.device)

    @staticmethod
    def _probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        x = logits.float()
        if params.temperature <= 1e-6:
            return F.one_hot(torch.argmax(x), x.shape[-1]).float()
        x = x / params.temperature
        x = min_p_filter(x, params.min_p)
        x = top_k_top_p_filter(x, params.top_k, params.top_p)
        return F.softmax(x, dim=-1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt_ids: List[int], params: Optional[SamplingParams] = None,
                 num_spec_tokens: int = 4, max_new_tokens: int = 64,
                 seed: Optional[int] = None) -> Dict:
        params = params or SamplingParams(max_tokens=max_new_tokens)
        if seed is not None:
            # torch.rand 只接受 CPU generator；而 torch.multinomial 要求 generator
            # 与张量同设备。两者诉求冲突，因此维护两个（用同一个种子初始化）。
            self.generator = torch.Generator().manual_seed(seed)
            self.dev_generator = (torch.Generator(device=self.device).manual_seed(seed)
                                  if self.device.type == "cuda" else self.generator)
        else:
            self.generator = None
            self.dev_generator = None
        t0 = time.time()

        max_len = len(prompt_ids) + max_new_tokens + num_spec_tokens + 8
        t_cache = self._new_cache(self.target, 1, max_len)
        d_cache = self._new_cache(self.draft, 1, max_len)

        x = torch.tensor([prompt_ids], device=self.device)
        logits = self.target(x, cache=t_cache)
        t_cache.advance(len(prompt_ids))
        self.draft(x, cache=d_cache)
        d_cache.advance(len(prompt_ids))
        stats = SpecDecodeStats(target_forwards=1)

        p = self._probs(logits[0, -1], params)
        first = self._sample_from(p)
        outputs = [first]
        # 同步：两个缓存都写入第一个 token
        self._sync([first], t_cache, d_cache, len(prompt_ids))
        stats.target_forwards += 1
        stats.draft_forwards += 1

        cached = len(prompt_ids) + 1
        while len(outputs) < max_new_tokens:
            # ---------- 1) draft 猜 γ 个 ----------
            # 注意位置对齐：要得到"位置 cached+j 的分布"，必须 forward 位置 cached+j-1 的 token。
            # 因此第一轮要先重放最后一个已知 token（位于 cached-1）。
            drafts, q_probs = [], []
            cur = outputs[-1]
            d_cache.write_pos = cached - 1
            for j in range(num_spec_tokens):
                xi = torch.tensor([[cur]], device=self.device)
                lg = self.draft(xi, positions=torch.tensor([[cached - 1 + j]], device=self.device),
                                cache=d_cache)
                d_cache.advance(1)
                q = self._probs(lg[0, -1], params)
                nxt = self._sample_from(q)
                drafts.append(nxt)
                q_probs.append(q)
                cur = nxt
                stats.draft_forwards += 1
            stats.draft_tokens += len(drafts)

            # ---------- 2) target 一次前向验证 ----------
            # 喂 [最后一个已知 token, d_1..d_γ]（共 γ+1 个）：
            # logits[i] 恰好是"位置 cached+i 的分布" → 用于验证 d_{i+1}；logits[γ] 给 bonus。
            seq = torch.tensor([[outputs[-1]] + drafts], device=self.device)
            pos = torch.arange(cached - 1, cached + len(drafts), device=self.device).unsqueeze(0)
            t_cache.write_pos = cached - 1
            t_logits = self.target(seq, positions=pos, cache=t_cache)
            t_cache.advance(len(drafts) + 1)
            stats.target_forwards += 1

            # ---------- 3) 接受-拒绝 ----------
            accepted = 0                 # 本轮最终写入的 token 数（含重采样/bonus）
            accepted_draft = 0           # 其中"被接受的 draft"数量 → 用于统计接受率
            rejected = False
            for i, d_tok in enumerate(drafts):
                p = self._probs(t_logits[0, i], params)
                q = q_probs[i]
                pd, qd = p[d_tok].item(), q[d_tok].item()
                if qd <= 0 or self._uniform() < min(1.0, pd / max(qd, 1e-9)):
                    outputs.append(d_tok)
                    accepted += 1
                    accepted_draft += 1
                else:
                    # 从 (p - q)_+ 重采样 —— 这一步保证了整体分布严格等于 target
                    resid = torch.clamp(p - q, min=0.0)
                    resid = resid / resid.sum().clamp(min=1e-9)
                    outputs.append(self._sample_from(resid))
                    accepted += 1
                    rejected = True
                    break
            if not rejected:
                # γ 个全被接受 → 再免费拿一个 bonus token（来自第 γ 个位置的分布）
                bonus = self._sample_from(self._probs(t_logits[0, -1], params))
                outputs.append(bonus)
                accepted += 1
                # accepted_draft 已在逐个接受时累加过了（此处为 γ），不重复计数
            stats.accepted_tokens += accepted_draft
            stats.rounds += 1

            # ---------- 4) 状态同步：把最后一个 token 写回两个缓存 ----------
            k = accepted
            pos_w = cached + k - 1
            t_cache.write_pos = pos_w
            d_cache.write_pos = pos_w
            self._sync([outputs[-1]], t_cache, d_cache, pos_w)
            stats.target_forwards += 1
            stats.draft_forwards += 1
            cached = pos_w + 1

            if outputs[-1] in params.stop_token_ids and not params.ignore_eos:
                break

        stats.elapsed_ms = (time.time() - t0) * 1000
        return {"output_ids": outputs, "text": self.tokenizer.decode(outputs) if self.tokenizer else "",
                "stats": stats}

    # ------------------------------------------------------------------ #
    def _sync(self, tokens: List[int], t_cache: DenseKVCache, d_cache: DenseKVCache, pos: int) -> None:
        """把 token 写进两个模型的缓存（保持两者状态一致）。"""
        x = torch.tensor([tokens], device=self.device)
        p = torch.arange(pos, pos + len(tokens), device=self.device).unsqueeze(0)
        self.target(x, positions=p, cache=t_cache)
        t_cache.advance(len(tokens))
        self.draft(x, positions=p, cache=d_cache)
        d_cache.advance(len(tokens))

    def _uniform(self) -> float:
        if self.generator is not None:
            return float(torch.rand(1, generator=self.generator).item())
        return float(torch.rand(1).item())

    def _sample_from(self, probs: torch.Tensor) -> int:
        g = self.dev_generator
        if g is not None and g.device.type == probs.device.type:
            return int(torch.multinomial(probs, 1, generator=g).item())
        return int(torch.multinomial(probs, 1).item())
