"""能力评测 + 数据去污染 / 配比。"""

import pytest
import torch

from tiniestgpt.data.decontaminate import (Decontaminator, contamination_rate,
                                           decontaminate, ngrams)
from tiniestgpt.data.mixture import AnnealingSchedule, MixtureSpec, build_mixture
from tiniestgpt.eval import available_tasks, evaluate
from tiniestgpt.eval.harness import EvalConfig
from tiniestgpt.eval.quality import generation_quality, repetition_rate
from tiniestgpt.eval.tasks import TASKS, build_task
from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model


class ToyTok:
    eos_id, pad_id = 1, 0

    def encode(self, text, **kw):
        return [min(2 + (ord(c) % 60), 63) for c in text[:48]] or [2]

    def decode(self, ids, **kw):
        return "".join(chr(32 + (i % 90)) for i in ids)


def _model():
    cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                      max_seq_len=128)
    return build_model(cfg).float()


# --------------------------------------------------------------------------- #
#  评测
# --------------------------------------------------------------------------- #
def test_builtin_tasks_are_deterministic_and_finite():
    for name in available_tasks():
        ex = build_task(name, n=3, seed=0)
        assert len(ex) == 3
        for e in ex:
            assert "prompt" in e and "answer" in e
            if TASKS[name].kind == "loglikelihood":
                assert len(e["choices"]) >= 2 and e["answer"] in e["choices"]


def test_evaluate_produces_table_and_quality():
    rep = evaluate(_model(), ToyTok(), EvalConfig(tasks=["cloze"], n_examples=4,
                                                  device="cpu", max_new_tokens=4))
    assert len(rep.results) == 1
    r = rep.results[0]
    assert 0.0 <= r.score <= 1.0
    assert rep.mean_score >= 0.0
    assert "cloze" in rep.table()
    d = rep.as_dict()
    assert "results" in d and "mean_score" in d


def test_generation_task_runs():
    rep = evaluate(_model(), ToyTok(), EvalConfig(tasks=["arithmetic"], n_examples=2,
                                                  device="cpu", max_new_tokens=4))
    assert rep.results[0].kind == "generate"
    assert isinstance(rep.quality, dict)


def test_repetition_detector():
    assert repetition_rate("a b c d e f g h i j") == 0.0
    assert repetition_rate("a b c d " * 6) > 0.5
    q = generation_quality(["a b c", "x y z", ""])
    assert 0.0 <= q["repetition_rate"] <= 1.0
    assert q["empty_ratio"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
#  去污染
# --------------------------------------------------------------------------- #
def test_ngrams_and_contamination_removal():
    eval_text = "the quick brown fox jumps over the lazy dog near the river bank today"
    train = [
        "the quick brown fox jumps over the lazy dog near the river bank today",  # 完全命中
        "completely unrelated content about something else entirely different here",  # 干净
    ]
    kept, rep = decontaminate(train, [eval_text], n=5, hit_rate=0.5)
    assert len(kept) == 1 and "unrelated" in kept[0]
    assert rep.removed == 1
    assert rep.as_dict()["removed"] == 1


def test_decontaminator_incremental_and_rate():
    d = Decontaminator(n=4, hit_rate=0.3)
    n = d.add_reference(["alpha beta gamma delta epsilon zeta"])
    assert n > 0
    assert d.is_contaminated("alpha beta gamma delta epsilon zeta")
    rate = contamination_rate(["alpha beta gamma delta epsilon zeta", "x y z w"],
                              ["alpha beta gamma delta epsilon zeta"], n=4)
    assert rate["max"] > 0.9 and rate["flagged_ratio"] > 0


def test_short_documents_are_not_automatically_safe():
    """短文档也要贡献 n-gram，否则"写得越短越安全"会成为漏洞。"""
    assert ngrams("hello world", n=13) != set()


# --------------------------------------------------------------------------- #
#  数据配比
# --------------------------------------------------------------------------- #
def test_mixture_respects_weights():
    a = [{"text": f"a{i}"} for i in range(100)]
    b = [{"text": f"b{i}"} for i in range(100)]
    out = build_mixture([MixtureSpec("a", a, weight=3.0), MixtureSpec("b", b, weight=1.0)],
                        seed=0, shuffle=True)
    na = sum(1 for d in out if d["_mixture_source"] == "a")
    nb = sum(1 for d in out if d["_mixture_source"] == "b")
    assert na > nb, f"权重 3:1 未生效 (a={na}, b={nb})"


def test_mixture_handles_degenerate_weights():
    out = build_mixture([MixtureSpec("a", [{"t": 1}], weight=0.0),
                         MixtureSpec("b", [{"t": 2}], weight=0.0)], seed=0)
    assert len(out) == 2          # 全 0 权重退化为均匀，不应崩


def test_annealing_schedule_interpolates():
    sch = AnnealingSchedule(start_step=100, end_step=200,
                            base_weights={"web": 0.8, "wiki": 0.2},
                            final_weights={"web": 0.2, "wiki": 0.8})
    assert sch.weights_at(50)["wiki"] == 0.2
    assert abs(sch.weights_at(150)["wiki"] - 0.5) < 1e-6
    assert sch.weights_at(500)["wiki"] == 0.8
    assert "wiki" in sch.report(150)
