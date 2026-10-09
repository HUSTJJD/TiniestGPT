"""生成质量指标：发现"分数上看不出来"的退化。

一个模型可以 ppl 很好看、任务分数也不掉，但生成时**只会复读同一句话**——
这种退化在交叉熵里几乎不可见，却是后训练（尤其 RL）最常见的失败模式之一。

这里提供三个便宜且敏感的指标：

* ``repetition_rate`` —— 4-gram 里重复出现的比例（复读检测器）；
* ``distinct_1 / distinct_2`` —— 去重后的 unigram / bigram 占全部的比例（多样性）；
* ``mean_length`` —— 平均长度（RL 里长度爆炸 / 长度坍缩的第一信号）。

2026 年的共识是：**任务分数 + 质量指标 + 成本** 三者一起看，
只看分数会系统性地被 reward hacking 骗到。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = ["repetition_rate", "distinct_n", "generation_quality"]


def _words(text: str) -> List[str]:
    return text.split()


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repetition_rate(text: str, n: int = 4) -> float:
    """重复的 n-gram 占比：``1 - |unique ngrams| / |ngrams|``。"""
    gs = _ngrams(_words(text), n)
    if len(gs) < 2:
        return 0.0
    return 1.0 - len(set(gs)) / len(gs)


def distinct_n(text: str, n: int = 1) -> float:
    """``|unique n-grams| / |n-grams|``——越高越多样。"""
    gs = _ngrams(_words(text), n)
    if not gs:
        return 0.0
    return len(set(gs)) / len(gs)


def generation_quality(texts: Sequence[str]) -> Dict[str, float]:
    if not texts:
        return {}
    reps = [repetition_rate(t) for t in texts]
    d1 = [distinct_n(t, 1) for t in texts]
    d2 = [distinct_n(t, 2) for t in texts]
    lens = [len(_words(t)) for t in texts]
    return {
        "repetition_rate": sum(reps) / len(reps),
        "distinct_1": sum(d1) / len(d1),
        "distinct_2": sum(d2) / len(d2),
        "mean_length": sum(lens) / len(lens),
        "empty_ratio": sum(1 for t in texts if not t.strip()) / len(texts),
    }
