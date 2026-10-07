"""Prefill / Decode 解耦实验：把"互扰"量化出来。

动机（AIInfraGuide 路线 3.5）：
Prefill 是 compute-bound 的大矩阵乘，Decode 是 memory-bound 的小矩阵乘。
把两者混在同一个 batch 里，一个 8k token 的长 prefill 会独占 GPU 几百毫秒，
所有正在生成的请求被拖慢 —— **这就是尾延迟 P95 爆炸的来源。**

DistServe / Splitwise 的解法：把 prefill 和 decode 放到**不同的 GPU 池**。

本脚本用两个引擎实例模拟"两个池"，对比：

* ``mixed``：所有请求进同一个引擎（现状）
* ``disagg``：长 prompt 请求与短请求分别进两个引擎

并给出解耦引入的新成本 —— **KV Cache 迁移**：
一个 7B 模型 2048 长度的 KV 约 4GB，IB 200Gb/s（≈25GB/s）也要 ~160ms。

> ⚠️ 本脚本在**单进程内串行**推进两个引擎，所以只能量化"干扰"与"算力配比"，
> 不能模拟真实的网络传输与并行加速。真实系统请直接读 DistServe 论文。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
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


def _pct(v: List[float], p: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def _make_engine(model, tok, max_model_len: int, concurrency: int, device: str,
                 cuda: bool) -> InferenceEngine:
    cfg = EngineConfig(max_num_seqs=concurrency, block_size=16,
                       dtype="bf16" if cuda else "fp32", device=device,
                       max_model_len=max_model_len)
    eng = InferenceEngine(build_model(model.cfg), tok, cfg)
    eng.model.load_state_dict(model.state_dict())
    return eng


def _tpot_of(engine: InferenceEngine, short_ids: List[int], params: SamplingParams,
             other: Optional[InferenceEngine] = None) -> List[float]:
    """推进到全部结束，返回每个 short 请求的 TPOT 序列（ms）。"""
    token_times: Dict[int, List[float]] = {sid: [] for sid in short_ids}
    while engine.scheduler.has_unfinished() or (other is not None and other.scheduler.has_unfinished()):
        for eng in ((engine, other) if other is not None else (engine,)):
            if eng is None:
                continue
            for out in eng.step():
                if out.seq_id in token_times:
                    token_times[out.seq_id].append(time.perf_counter())
    tpots: List[float] = []
    for times in token_times.values():
        tpots.extend((times[i + 1] - times[i]) * 1000 for i in range(len(times) - 1))
    return tpots


def run(n_long: int, n_short: int, long_prompt: int, short_prompt: int,
        max_tokens: int, preset: str = "tiny", checkpoint: Optional[str] = None,
        tokenizer_path: Optional[str] = None, ib_gb_s: float = 25.0) -> Dict[str, float]:
    """两次真实运行，得到**干扰系数**与**解耦的盈亏平衡点**。

    为什么不直接"起两个引擎对比"：单进程里两个引擎只能**串行**推进，
    墙钟时间被"等另一个引擎"主导，测出来的解耦反而更慢 —— 那是测量假象，
    不是结论。所以这里改成：

    1. **baseline**：引擎里只有短请求 → 纯 decode 延迟（= 解耦后 Decode 池的理论值）
    2. **mixed**：长 prefill 与短请求混在同一引擎 → 短请求被拖慢多少
    3. 干扰系数 = mixed / baseline；解耦的收益 = 把这个系数压回 1
    4. 解耦的代价 = KV 迁移耗时 → 算出"生成多少 token 才回本"
    """
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    model = load_model(checkpoint, map_location="cpu") if checkpoint else build_model(preset)
    model = model.to(device).eval()

    tok = DummyTokenizer()
    if tokenizer_path:
        from tiniestgpt.data.tokenizer import Tokenizer

        tok = Tokenizer.load(tokenizer_path)

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    long_prompts = [list(range(4, 4 + long_prompt)) for _ in range(n_long)]
    short_prompts = [list(range(4, 4 + short_prompt)) for _ in range(n_short)]

    # ---------------- ① 纯 decode（解耦后 Decode 池的理想值） ----------------
    baseline = _make_engine(model, tok, short_prompt + max_tokens + 16, n_short, device, cuda)
    b_ids = [baseline.add_request(p, params) for p in short_prompts]
    base_tpot = _tpot_of(baseline, b_ids, params)

    # ---------------- ② 混合部署（现状） ----------------
    mixed = _make_engine(model, tok, long_prompt + max_tokens + 16,
                         n_long + n_short, device, cuda)
    for p in long_prompts:
        mixed.add_request(p, params)
    m_ids = [mixed.add_request(p, params) for p in short_prompts]
    mixed_tpot = _tpot_of(mixed, m_ids, params)

    b_p50, b_p95 = _pct(base_tpot, 50), _pct(base_tpot, 95)
    m_p50, m_p95 = _pct(mixed_tpot, 50), _pct(mixed_tpot, 95)

    # ---------------- KV 迁移成本与盈亏平衡 ----------------
    cfg = model.cfg
    n_kv_heads = cfg.n_kv_heads
    head_dim = cfg.head_dim or (cfg.dim // cfg.n_heads)
    kv_bytes = 2 * cfg.n_layers * n_kv_heads * head_dim * long_prompt * 2   # bf16
    transfer_ms = kv_bytes / (ib_gb_s * 1e9) * 1000
    saved_per_token_ms = max(m_p95 - b_p95, 0.0)
    break_even = transfer_ms / saved_per_token_ms if saved_per_token_ms > 1e-6 else float("inf")

    return {
        "baseline_tpot_p50_ms": b_p50,
        "baseline_tpot_p95_ms": b_p95,
        "mixed_tpot_p50_ms": m_p50,
        "mixed_tpot_p95_ms": m_p95,
        "interference_p50": m_p50 / max(b_p50, 1e-9),
        "interference_p95": m_p95 / max(b_p95, 1e-9),
        "kv_bytes_per_request": float(kv_bytes),
        "kv_transfer_ms": transfer_ms,
        "saved_ms_per_token": saved_per_token_ms,
        "break_even_tokens": break_even,
        "n_long": float(n_long),
        "n_short": float(n_short),
    }


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser("prefill/decode disaggregation")
    ap.add_argument("--n-long", type=int, default=4, help="长 prompt 请求数（prefill 重型）")
    ap.add_argument("--n-short", type=int, default=8, help="短请求数（decode 重型）")
    ap.add_argument("--long-prompt", type=int, default=512)
    ap.add_argument("--short-prompt", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--preset", type=str, default="tiny")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--ib-gb-s", type=float, default=25.0, help="跨机带宽（GB/s），IB 200Gb/s≈25")
    args = ap.parse_args(argv)

    r = run(args.n_long, args.n_short, args.long_prompt, args.short_prompt,
            args.max_tokens, args.preset, args.checkpoint, args.tokenizer, args.ib_gb_s)

    print("\n=== Prefill/Decode 干扰实验 ===")
    print(f"长请求 {int(r['n_long'])} × {args.long_prompt} token | "
          f"短请求 {int(r['n_short'])} × {args.short_prompt} token | "
          f"生成 {args.max_tokens} token")
    print(f"  {'':<16}{'P50 TPOT':>12}{'P95 TPOT':>12}")
    print(f"  纯 decode（理想） {r['baseline_tpot_p50_ms']:>9.2f}ms{r['baseline_tpot_p95_ms']:>10.2f}ms")
    print(f"  混合部署（现状）  {r['mixed_tpot_p50_ms']:>9.2f}ms{r['mixed_tpot_p95_ms']:>10.2f}ms")
    print(f"  干扰系数          {r['interference_p50']:>11.2f}×{r['interference_p95']:>11.2f}×")
    print("\n=== 解耦的代价与盈亏平衡 ===")
    print(f"  每请求 KV: {r['kv_bytes_per_request'] / 1024 ** 2:.2f} MB")
    print(f"  IB {args.ib_gb_s:.0f} GB/s 下迁移耗时: {r['kv_transfer_ms']:.2f} ms")
    print(f"  解耦每 token 能省: {r['saved_ms_per_token']:.2f} ms")
    be = r["break_even_tokens"]
    print(f"  盈亏平衡：单请求需生成 ≥ {'∞' if be == float('inf') else f'{be:.0f}'} 个 token 才回本")
    print("  → 短回答场景（< 盈亏平衡点）解耦反而更慢，这正是「不是所有业务都该解耦」的原因")
    print("\n注：单进程串行推进，只量化干扰与迁移成本，不模拟网络并行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
