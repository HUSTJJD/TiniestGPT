"""采样：把 logits 变成 token。

采样器是"模型性格"的来源，也是很多线上 bug 的根源。这里实现的处理器（按推荐顺序）：

1. **repetition / frequency / presence penalty**：压住复读。
   注意 frequency 用「出现次数」、presence 只用「是否出现过」（二值），
   且标准实现会把**已出现过的 logit 为负**的惩罚得更狠（保持符号一致）。
2. **temperature**：整体缩放；T→0 等价于 greedy。
3. **min-p**（2024 年流行）：``p ≥ min_p · p_max`` 的候选集合自适应大小，
   比 top-p 更能兼顾"确定时更确定、不确定时更多样"。
4. **top-k / top-p (nucleus)**：截断长尾。
5. **typical-p**：按「信息量接近条件熵」选 token，抑制过度常见或过度罕见的输出。

另外提供 **beam search** 与 **best-of-n** 两种搜索式解码的参考实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["SamplingParams", "Sampler", "apply_penalties", "min_p_filter", "top_k_top_p_filter"]


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0                 # 0 = 关闭
    top_p: float = 1.0             # 1.0 = 关闭
    min_p: float = 0.0             # 0 = 关闭
    typical_p: float = 1.0         # 1.0 = 关闭
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 128
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None
    n: int = 1                     # best-of-n / beam 的候选数（引擎层用）
    # 结构化输出：给定时引擎会开启约束解码，保证输出是合法 JSON
    # （OpenAI API 里对应 response_format={"type": "json_object"}）
    json_schema: Optional[dict] = None

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if not (0 < self.top_p <= 1.0):
            raise ValueError("top_p 必须在 (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k 必须 >= 0")


def apply_penalties(logits: torch.Tensor,                 # [V]
                    prompt_ids: Sequence[int],
                    output_ids: Sequence[int],
                    repetition_penalty: float = 1.0,
                    frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0) -> torch.Tensor:
    """按已出现的 token 调整 logits（保持原符号，避免把负 logit 翻正）。"""
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits
    counts: Dict[int, int] = {}
    for t in list(prompt_ids) + list(output_ids):
        counts[int(t)] = counts.get(int(t), 0) + 1
    if not counts:
        return logits
    idx = torch.tensor(sorted(counts), device=logits.device, dtype=torch.long)
    cnt = torch.tensor([counts[int(i)] for i in idx.tolist()],
                       device=logits.device, dtype=logits.dtype)

    penalty = torch.zeros_like(logits)
    if repetition_penalty != 1.0:
        rp = torch.full_like(cnt, repetition_penalty)
        # 负 logit 乘以 penalty 会更负；标准做法是保持符号
        penalty[idx] = torch.where(logits[idx] > 0, logits[idx] / rp, logits[idx] * rp) - logits[idx]
    if frequency_penalty != 0.0:
        penalty[idx] -= frequency_penalty * cnt
    if presence_penalty != 0.0:
        penalty[idx] -= presence_penalty * (cnt > 0).to(logits.dtype)
    return logits + penalty


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0:
        return logits
    probs = F.softmax(logits.float(), dim=-1)
    top = probs.max()
    keep = probs >= (min_p * top)
    return logits.masked_fill(~keep, torch.finfo(logits.dtype).min)


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals, _ = torch.topk(logits, k)
        threshold = vals[..., -1]
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits.float(), dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # 去掉"累积概率已超过 top_p"的部分（保留第一个超阈值的 token）
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


class Sampler:
    """无状态采样器（随机源由外部注入，便于复现）。"""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def _generator(self, params: SamplingParams) -> Optional[torch.Generator]:
        if params.seed is None:
            return None
        g = torch.Generator(device=self.device.type if self.device.type == "cuda" else "cpu")
        g.manual_seed(params.seed)
        return g

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, params: SamplingParams,
               prompt_ids: Sequence[int] = (), output_ids: Sequence[int] = (),
               generator: Optional[torch.Generator] = None) -> int:
        """从单条 logits [V] 采样一个 token。"""
        logits = logits.float()
        logits = apply_penalties(logits, prompt_ids, output_ids,
                                 params.repetition_penalty, params.frequency_penalty,
                                 params.presence_penalty)
        if params.temperature <= 1e-6:
            return int(torch.argmax(logits).item())
        logits = logits / params.temperature
        logits = min_p_filter(logits, params.min_p)
        logits = top_k_top_p_filter(logits, params.top_k, params.top_p)
        if params.typical_p < 1.0:
            logits = self._typical_filter(logits, params.typical_p)
        probs = F.softmax(logits, dim=-1)
        if generator is not None:
            idx = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            idx = torch.multinomial(probs, num_samples=1)
        return int(idx.item())

    @staticmethod
    def _typical_filter(logits: torch.Tensor, typical_p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        # 信息量 -log p 与熵的偏离程度越小越"典型"
        dev = (torch.log(probs + 1e-9) + entropy).abs()
        sorted_dev, sorted_idx = torch.sort(dev)
        cum = torch.cumsum(F.softmax(-sorted_dev.float(), dim=-1), dim=-1)
        remove = cum - F.softmax(-sorted_dev.float(), dim=-1) > typical_p
        mask = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, sorted_idx, remove)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def beam_search(self, model, input_ids: torch.Tensor, beam_width: int = 4,
                    max_new_tokens: int = 32, eos_id: Optional[int] = None,
                    length_penalty: float = 1.0) -> List[List[int]]:
        """教学用 beam search（逐 token 前向，不复用 KV，慢但清晰）。"""
        B = input_ids.shape[0]
        beams = [(input_ids[b].tolist(), 0.0) for b in range(B)]
        finished: List[List[int]] = []
        for _ in range(max_new_tokens):
            cands = []
            for seq, score in beams:
                x = torch.tensor([seq], device=input_ids.device)
                logits = model(x)[:, -1, :]
                logp = torch.log_softmax(logits.float(), dim=-1)[0]
                topv, topi = torch.topk(logp, beam_width)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    cands.append((seq + [i], score + v))
            cands.sort(key=lambda t: t[1] / (len(t[0]) ** length_penalty), reverse=True)
            beams = []
            for seq, sc in cands:
                if eos_id is not None and seq[-1] == eos_id:
                    finished.append(seq)
                else:
                    beams.append((seq, sc))
                if len(beams) >= beam_width:
                    break
            if not beams:
                break
        return finished if finished else [b[0] for b in beams]
