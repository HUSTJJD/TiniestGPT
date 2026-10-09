"""评测执行器：把"模型 + tokenizer + 任务"变成一张分数表。

两种打分方式，对应两类能力：

* **log-likelihood 打分**（完形 / 选择题）——把每个候选接在 prompt 后面，
  算 ``Σ log p(token)``，取最大者。它**不受解码质量影响**，
  是"模型到底有没有学到这个知识"的干净度量。
* **生成 + 规则判定**——greedy 解码后用规则/解析器判定，
  这正是 2026 年 RLVR（可验证奖励）的标准打分方式：
  **答案对不对由规则说了算，不需要另一个模型来当裁判。**

另外提供 :func:`generation_quality`：重复率 / distinct-n，
用来发现"模型没崩但只会复读"这种分数上看不出来的退化。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .quality import generation_quality

__all__ = ["EvalConfig", "TaskResult", "EvalReport", "greedy_generate",
           "score_loglikelihood", "evaluate"]


@dataclass
class EvalConfig:
    tasks: List[str] = field(default_factory=lambda: ["arithmetic", "cloze"])
    n_examples: int = 32
    seed: int = 0
    max_new_tokens: int = 32
    batch_size: int = 8
    use_cache: bool = True           # decode 时是否用 KV Cache（混合架构会用递归状态）
    length_normalized: bool = False  # loglikelihood 是否按 token 数归一（避免选项长度偏差）
    device: str = "auto"
    dtype: str = "float32"


@dataclass
class TaskResult:
    name: str
    kind: str
    n: int
    score: float
    correct: int
    latency_ms: float
    samples: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"task": self.name, "kind": self.kind, "n": self.n,
                "score": round(self.score, 4), "correct": self.correct,
                "latency_ms": round(self.latency_ms, 1)}


@dataclass
class EvalReport:
    results: List[TaskResult]
    quality: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    # ------------------------------------------------------------------ #
    def table(self) -> str:
        lines = ["%-12s %-14s %6s %8s %8s" % ("task", "kind", "n", "score", "correct")]
        lines.append("-" * 54)
        for r in self.results:
            lines.append("%-12s %-14s %6d %8.3f %8d" % (r.name, r.kind, r.n, r.score, r.correct))
        if self.quality:
            lines.append("-" * 54)
            lines.append("generation quality: " + ", ".join(
                f"{k}={v:.3f}" for k, v in self.quality.items()))
        lines.append(f"total: {self.total_ms:.0f} ms")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {"results": [r.as_dict() for r in self.results],
                "quality": self.quality, "total_ms": round(self.total_ms, 1),
                "mean_score": self.mean_score}

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / max(len(self.results), 1)

    def samples(self, limit: int = 2) -> str:
        out = []
        for r in self.results:
            for s in r.samples[:limit]:
                out.append(f"[{r.name}] {s}")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
#  解码
# --------------------------------------------------------------------------- #
def greedy_generate(model, tokenizer, prompt_ids: Sequence[int], max_new_tokens: int,
                    device: torch.device, use_cache: bool = True,
                    stop_strings: Sequence[str] = ()) -> str:
    """greedy 解码。混合架构（GDN / Mamba）会自动走递归状态路径。"""
    model.eval()
    ids = list(prompt_ids)
    if not ids:
        ids = [0]
    x = torch.tensor([ids], device=device)
    cache = None
    if use_cache:
        try:
            from ..model.kv_cache import DenseKVCache

            h, d = model.cache_spec()
            cache = DenseKVCache(model.cfg.n_layers, 1, len(ids) + max_new_tokens + 8,
                                 h, d, dtype=torch.float32, device=device)
        except Exception:
            cache = None
    generated: List[int] = []
    with torch.no_grad():
        logits = model(x, cache=cache)
        if cache is not None:
            cache.advance(len(ids))
        for _ in range(max_new_tokens):
            nxt = int(torch.argmax(logits[0, -1]).item())
            generated.append(nxt)
            if nxt in (getattr(tokenizer, "eos_id", -1), getattr(tokenizer, "pad_id", -1)):
                break
            text = tokenizer.decode(generated)
            if any(s in text for s in stop_strings):
                break
            x = torch.tensor([[nxt]], device=device)
            logits = model(x, cache=cache)
            if cache is not None:
                cache.advance(1)
    return tokenizer.decode(generated)


def score_loglikelihood(model, tokenizer, prompt_ids: Sequence[int],
                        choice_ids: Sequence[int], device: torch.device,
                        length_normalized: bool = False) -> float:
    """``Σ log p(choice | prompt)``（可选按 token 数归一）。"""
    ids = list(prompt_ids) + list(choice_ids)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(x)[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    start = len(prompt_ids) - 1
    total = 0.0
    for i, tid in enumerate(choice_ids):
        total += float(lp[start + i, tid].item())
    return total / max(len(choice_ids), 1) if length_normalized else total


# --------------------------------------------------------------------------- #
def evaluate(model, tokenizer, cfg: Optional[EvalConfig] = None) -> EvalReport:
    from .tasks import TASKS

    cfg = cfg or EvalConfig()
    device_str = cfg.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = model.to(device).eval()

    results: List[TaskResult] = []
    all_outputs: List[str] = []
    t_start = time.time()

    for name in cfg.tasks:
        task = TASKS[name]
        examples = task.build(cfg.n_examples, cfg.seed)
        t0 = time.time()
        correct = 0.0
        samples: List[Dict[str, str]] = []

        for ex in examples:
            if task.kind == "loglikelihood":
                p_ids = tokenizer.encode(ex["prompt"])
                best, best_txt = None, ""
                for ch in ex["choices"]:
                    s = score_loglikelihood(model, tokenizer, p_ids, tokenizer.encode(" " + ch),
                                            device, cfg.length_normalized)
                    if best is None or s > best:
                        best, best_txt = s, ch
                hit = task.scorer(best_txt, ex)
                correct += hit
                if len(samples) < 2:
                    samples.append(f"gold={ex['answer']} pred={best_txt}")
            else:
                p_ids = tokenizer.encode(ex["prompt"])
                out = greedy_generate(model, tokenizer, p_ids, cfg.max_new_tokens,
                                      device, cfg.use_cache, task.stop)
                all_outputs.append(out)
                hit = task.scorer(out, ex)
                correct += hit
                if len(samples) < 2:
                    samples.append(f"gold={ex['answer']} pred={out.strip()[:40]!r}")

        n = len(examples)
        results.append(TaskResult(name=name, kind=task.kind, n=n,
                                  score=correct / max(n, 1), correct=int(correct),
                                  latency_ms=(time.time() - t0) * 1000, samples=samples))

    quality = generation_quality(all_outputs) if all_outputs else {}
    return EvalReport(results=results, quality=quality,
                      total_ms=(time.time() - t_start) * 1000)
