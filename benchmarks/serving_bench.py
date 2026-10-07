"""服务化基准：输出**六项指标**，并支持 CI 回归门禁。

对齐 AIInfraGuide 路线 3.6「性能分析与 Benchmark」的检验标准——
每次评测至少要有 6 个指标，缺任何一个都可能漏掉瓶颈：

1. **QPS**（完成请求数 / 秒）
2. **TTFT P50 / P95**（首 token 延迟）
3. **TPOT P50 / P95**（每 token 延迟，决定"卡不卡"）
4. **端到端吞吐** token/s
5. **显存峰值** MB
6. **GPU 繁忙率**（step 耗时 / 墙钟，用来看调度有没有把 GPU 喂饱）

外加 **goodput**：满足 SLO（默认 TTFT<500ms 且 TPOT<50ms）的请求占比。
raw QPS 高不等于用户体验好 —— 这正是 goodput 存在的意义。

用法::

    python benchmarks/serving_bench.py --num-requests 64 --concurrency 16
    python benchmarks/serving_bench.py --checkpoint out/tiny/last.pt --tokenizer data/tokenizer.json
    # 回归门禁：与基线对比，退化超过阈值就返回非零退出码（可直接接 CI）
    python benchmarks/serving_bench.py --baseline out/serving_baseline.json --max-regression 0.05
    python benchmarks/serving_bench.py --save-baseline out/serving_baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tiniestgpt.cli import _force_utf8_stdio
from tiniestgpt.inference.engine import EngineConfig, InferenceEngine
from tiniestgpt.inference.sampler import SamplingParams
from tiniestgpt.model.factory import build_model, load_model


class DummyTokenizer:
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        return [4 + (i % 60) for i in range(len(text))][:256] or [4]

    def decode(self, ids, skip_special=True):
        return "".join(chr(40 + int(i) % 50) for i in ids)


def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def run_bench(num_requests: int, concurrency: int, prompt_len: int, max_tokens: int,
              preset: str = "tiny", checkpoint: Optional[str] = None,
              tokenizer_path: Optional[str] = None, block_size: int = 16,
              ttft_slo_ms: float = 500.0, tpot_slo_ms: float = 50.0,
              prefix_cache: bool = True) -> Dict[str, float]:
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    model = load_model(checkpoint, map_location="cpu") if checkpoint else build_model(preset)
    model = model.to(device).eval()

    tok = DummyTokenizer()
    if tokenizer_path:
        from tiniestgpt.data.tokenizer import Tokenizer

        tok = Tokenizer.load(tokenizer_path)

    cfg = EngineConfig(max_num_seqs=concurrency, block_size=block_size,
                       dtype="bf16" if cuda else "fp32", device=device,
                       enable_prefix_caching=prefix_cache,
                       max_model_len=prompt_len + max_tokens + 16)
    engine = InferenceEngine(build_model(model.cfg), tok, cfg)
    engine.model.load_state_dict(model.state_dict())

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    prompts = [list(range(4, 4 + prompt_len)) for _ in range(num_requests)]

    # ---- 记录每条请求的时间线 ----
    arrival: Dict[int, float] = {}
    first_token_at: Dict[int, float] = {}
    token_times: Dict[int, List[float]] = {}
    finished_at: Dict[int, float] = {}
    pending = list(range(num_requests))
    in_flight: Dict[int, int] = {}          # seq_id -> request index

    def admit() -> None:
        while pending and len(in_flight) < concurrency:
            rid = pending.pop(0)
            sid = engine.add_request(prompts[rid], params)
            in_flight[sid] = rid
            arrival[sid] = time.perf_counter()
            token_times[sid] = []

    admit()
    step_ms: List[float] = []
    t_start = time.perf_counter()

    while engine.scheduler.has_unfinished() or pending:
        t0 = time.perf_counter()
        outs = engine.step()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        now = time.perf_counter()
        for out in outs:
            sid = out.seq_id
            n = out.completion_tokens
            if n >= 1 and sid not in first_token_at:
                first_token_at[sid] = now
            while len(token_times[sid]) < n:
                token_times[sid].append(now)
            if out.finished:
                finished_at[sid] = now
                in_flight.pop(sid, None)
        admit()
        if not outs and not pending and not engine.scheduler.has_unfinished():
            break

    wall_s = time.perf_counter() - t_start
    if cuda:
        torch.cuda.synchronize()

    # ---- 指标 ----
    ttft = [(first_token_at[s] - arrival[s]) * 1000 for s in first_token_at]
    tpot: List[float] = []
    for s, times in token_times.items():
        if len(times) < 2:
            continue
        tpot.extend([(times[i + 1] - times[i]) * 1000 for i in range(len(times) - 1)])
    total_tokens = sum(len(v) for v in token_times.values())
    good = sum(1 for s in first_token_at
               if (first_token_at[s] - arrival[s]) * 1000 <= ttft_slo_ms
               and (statistics.fmean([(token_times[s][i + 1] - token_times[s][i]) * 1000
                                      for i in range(len(token_times[s]) - 1)])
                    if len(token_times[s]) > 1 else 0.0) <= tpot_slo_ms)

    n_done = len(finished_at)
    return {
        "num_requests": float(num_requests),
        "concurrency": float(concurrency),
        "qps": n_done / wall_s if wall_s > 0 else float("nan"),
        "ttft_p50_ms": _pct(ttft, 50),
        "ttft_p95_ms": _pct(ttft, 95),
        "tpot_p50_ms": _pct(tpot, 50),
        "tpot_p95_ms": _pct(tpot, 95),
        "throughput_tokens_per_s": total_tokens / wall_s if wall_s > 0 else float("nan"),
        "peak_memory_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2) if cuda else 0.0,
        "gpu_busy_ratio": (sum(step_ms) / 1000.0 / wall_s) if wall_s > 0 else float("nan"),
        "goodput": good / max(n_done, 1),
        "wall_s": wall_s,
        "completed": float(n_done),
        "steps": float(len(step_ms)),
    }


_HEADERS = [
    ("qps", "QPS", "{:.2f}"),
    ("ttft_p50_ms", "TTFT P50 (ms)", "{:.1f}"),
    ("ttft_p95_ms", "TTFT P95 (ms)", "{:.1f}"),
    ("tpot_p50_ms", "TPOT P50 (ms)", "{:.2f}"),
    ("tpot_p95_ms", "TPOT P95 (ms)", "{:.2f}"),
    ("throughput_tokens_per_s", "吞吐 (tok/s)", "{:.1f}"),
    ("peak_memory_mb", "显存峰值 (MB)", "{:.0f}"),
    ("gpu_busy_ratio", "GPU 繁忙率", "{:.1%}"),
    ("goodput", "Goodput", "{:.1%}"),
]


def print_table(m: Dict[str, float]) -> None:
    print("\n=== 服务化基准 ===")
    print(f"请求数 {int(m['num_requests'])}  并发 {int(m['concurrency'])}  "
          f"墙钟 {m['wall_s']:.2f}s  步数 {int(m['steps'])}")
    for key, label, fmt in _HEADERS:
        print(f"  {label:<16} {fmt.format(m[key])}")


def compare(m: Dict[str, float], base: Dict[str, float], max_reg: float) -> int:
    """回归门禁：吞吐下降 / 延迟上升超过阈值 → 返回 1。"""
    # 越大越好的指标
    higher = ["qps", "throughput_tokens_per_s", "goodput"]
    # 越小越好的指标
    lower = ["ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "peak_memory_mb"]
    bad = []
    for k in higher:
        b, v = base.get(k), m.get(k)
        if b and v == v and b == b and b > 0:
            drop = (b - v) / b
            if drop > max_reg:
                bad.append(f"{k}: {b:.3f} → {v:.3f}（下降 {drop:.1%}）")
    for k in lower:
        b, v = base.get(k), m.get(k)
        if b and v == v and b == b and b > 0:
            rise = (v - b) / b
            if rise > max_reg:
                bad.append(f"{k}: {b:.3f} → {v:.3f}（上升 {rise:.1%}）")
    print("\n=== 回归门禁 ===")
    if not bad:
        print(f"  通过（阈值 {max_reg:.0%}）")
        return 0
    for line in bad:
        print("  退化:", line)
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser("serving benchmark")
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--preset", type=str, default="tiny")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--save-baseline", type=str, default=None)
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--max-regression", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    m = run_bench(args.num_requests, args.concurrency, args.prompt_len, args.max_tokens,
                  args.preset, args.checkpoint, args.tokenizer,
                  prefix_cache=not args.no_prefix_cache)

    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print_table(m)

    if args.save_baseline:
        Path(args.save_baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_baseline).write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"\n基线已保存: {args.save_baseline}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        return compare(m, base, args.max_regression)
    return 0


if __name__ == "__main__":
    sys.exit(main())
