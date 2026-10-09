"""能力评测包：``tiniestgpt.eval``

2026 年一个"全链路"项目如果只有 ppl 和吞吐，是没法回答
**"模型到底学会了没有"** 的。这个包补上那一环：

* :mod:`tasks`   —— 离线合成的可验证任务（算术 / 长程召回 / JSON / 完形）
* :mod:`harness` —— 两种打分：log-likelihood 与 生成+规则判定
* :mod:`quality` —— 生成质量指标（复读率 / 多样性 / 长度）

用法::

    uv run python -m tiniestgpt.cli eval --checkpoint out/tiny/last.pt \
        --tokenizer data/tokenizer.json --tasks arithmetic cloze json
"""

from __future__ import annotations

from .harness import EvalConfig, EvalReport, TaskResult, evaluate, greedy_generate, score_loglikelihood
from .quality import distinct_n, generation_quality, repetition_rate
from .tasks import TASKS, available_tasks, build_task

__all__ = [
    "TASKS", "available_tasks", "build_task",
    "EvalConfig", "EvalReport", "TaskResult", "evaluate",
    "greedy_generate", "score_loglikelihood",
    "generation_quality", "repetition_rate", "distinct_n",
]
