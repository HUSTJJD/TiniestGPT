"""质量回归门禁：**能力**门禁，而不只是性能门禁。

``benchmarks/serving_bench.py`` 守的是吞吐/延迟（性能门禁），
这个脚本守的是"模型没被改傻"（质量门禁）。

为什么必须两个都有：性能优化（量化、内核替换、架构改造）和
后训练（RL、蒸馏）都可能在**吞吐上升的同时悄悄把模型改坏**。
只有性能门禁的话，一次"成功"的量化可能把准确率砍掉 20% 而没人发现。

用法::

    # 1) 先建立基线
    uv run python benchmarks/quality_gate.py --checkpoint out/tiny/last.pt --save-baseline

    # 2) 之后每次改动都跑一遍（CI 里用它拦截退化）
    uv run python benchmarks/quality_gate.py --checkpoint out/new/last.pt --compare

判定规则（都可配）：

* **绝对下限**：单任务分数不得低于 ``--min-score``（防"某个能力直接归零"）；
* **相对退化**：相对基线的下降不得超过 ``--max-drop``（防"整体悄悄变差"）；
* **质量指标**：复读率不得超过 ``--max-repetition``（防 RL 训出复读机）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from tiniestgpt.common.logging import get_logger

log = get_logger("tiniestgpt.bench")

DEFAULT_BASELINE = Path("out/quality_baseline.json")


def _force_utf8_stdio() -> None:
    for s in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def run_eval(checkpoint: str | None, preset: str, tokenizer: str, tasks: List[str],
             n: int, seed: int, device: str) -> Dict[str, Any]:
    from tiniestgpt.data.tokenizer import Tokenizer
    from tiniestgpt.eval import EvalConfig, available_tasks, evaluate
    from tiniestgpt.model.factory import build_model, load_model

    tok = Tokenizer.load(tokenizer)
    if checkpoint and Path(checkpoint).exists():
        model = load_model(checkpoint, map_location="cpu")
    else:
        # 没有 checkpoint 时按种子固定初始化，否则两次运行分数不同、门禁毫无意义
        torch.manual_seed(seed)
        model = build_model(preset)
    cfg = EvalConfig(tasks=tasks or available_tasks(), n_examples=n, seed=seed, device=device)
    rep = evaluate(model, tok, cfg)
    return rep.as_dict()


def check(report: Dict[str, Any], baseline: Dict[str, Any] | None,
          min_score: float, max_drop: float, max_repetition: float) -> List[str]:
    """返回违规列表（空 = 通过）。"""
    problems: List[str] = []
    base_scores = {r["task"]: r["score"] for r in baseline.get("results", [])} if baseline else {}

    for r in report["results"]:
        name, score = r["task"], r["score"]
        if score < min_score:
            problems.append(f"{name}: 分数 {score:.3f} 低于绝对下限 {min_score:.3f}")
        if name in base_scores:
            drop = base_scores[name] - score
            if drop > max_drop:
                problems.append(
                    f"{name}: 相对基线下降 {drop:.3f}（{base_scores[name]:.3f} → {score:.3f}），"
                    f"超过允许 {max_drop:.3f}")
    q = report.get("quality") or {}
    if q.get("repetition_rate", 0.0) > max_repetition:
        problems.append(f"复读率 {q['repetition_rate']:.3f} 超过上限 {max_repetition:.3f}")
    return problems


def main() -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser("quality gate")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--preset", type=str, default="tiny")
    ap.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--n-examples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE))
    ap.add_argument("--save-baseline", action="store_true", help="把当前结果存为基线")
    ap.add_argument("--compare", action="store_true", help="与基线对比并做门禁判定")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-drop", type=float, default=0.05)
    ap.add_argument("--max-repetition", type=float, default=0.9)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    report = run_eval(args.checkpoint, args.preset, args.tokenizer, args.tasks,
                      args.n_examples, args.seed, args.device)

    print("\n=== 能力评测 ===")
    print("%-12s %6s %8s" % ("task", "n", "score"))
    for r in report["results"]:
        print("%-12s %6d %8.3f" % (r["task"], r["n"], r["score"]))
    if report.get("quality"):
        print("quality: " + ", ".join(f"{k}={v:.3f}" for k, v in report["quality"].items()))
    print(f"mean_score: {report['mean_score']:.4f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    baseline_path = Path(args.baseline)
    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[quality-gate] 基线已写入 {baseline_path}")
        return 0

    if args.compare:
        if not baseline_path.exists():
            print(f"\n[quality-gate] 没有基线 {baseline_path}；"
                  f"先跑一次 --save-baseline", file=sys.stderr)
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        problems = check(report, baseline, args.min_score, args.max_drop, args.max_repetition)
        if problems:
            print("\n[quality-gate] ❌ 未通过:")
            for p in problems:
                print("  - " + p)
            return 1
        print("\n[quality-gate] ✅ 通过：无能力退化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
