"""推理优化消融实验：**逐项打开优化，记录吞吐变化**。

这是理解 AI Infra 最有效的方法——每个优化到底值多少，用数字说话::

    python benchmarks/inference_ablation.py                     # 用随机初始化模型
    python benchmarks/inference_ablation.py --checkpoint ck.pt \  # 用真实权重
           --tokenizer data/tokenizer.json --batch 8 --prompt-len 128 --gen-len 64

输出一张表：优化项 / 吞吐(tok/s) / 相对基线加速 / 显存(MB)。
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import torch

from tiniestgpt.common.logging import get_logger
from tiniestgpt.cli import _force_utf8_stdio
from tiniestgpt.inference.engine import EngineConfig, InferenceEngine
from tiniestgpt.inference.quantization import QuantizedPagedKVCache
from tiniestgpt.inference.sampler import SamplingParams
from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model, load_model
from tiniestgpt.model.kv_cache import DenseKVCache

log = get_logger("bench")


class DummyTokenizer:
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        return [4 + (i % 60) for i in range(len(text))][:256] or [4]

    def decode(self, ids, skip_special=True):
        return "".join(chr(40 + int(i) % 50) for i in ids)


def _mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


# --------------------------------------------------------------------------- #
# 各级实现
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_no_cache(model, prompts: List[List[int]], gen_len: int, device: torch.device) -> float:
    """L0：每步重算整个序列（没有 KV Cache）——O(L²) 的教科书式实现。"""
    model.eval()
    dev = torch.device(device)
    t0 = time.time()
    n = 0
    for p in prompts:
        ids = list(p)
        for _ in range(gen_len):
            x = torch.tensor([ids], device=dev)
            logits = model(x)
            ids.append(int(logits[0, -1].argmax()))
            n += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t0)


@torch.no_grad()
def bench_dense_cache(model, prompts: List[List[int]], gen_len: int, max_len: int,
                      device: torch.device, dtype: torch.dtype) -> float:
    """L1：稠密 KV Cache（单序列，逐条生成）。"""
    model.eval()
    dev = torch.device(device)
    h, d = model.cache_spec()
    t0 = time.time()
    n = 0
    for p in prompts:
        cache = DenseKVCache(model.cfg.n_layers, 1, max_len, h, d, dtype=dtype, device=dev)
        x = torch.tensor([p], device=dev)
        logits = model(x, cache=cache)
        cache.advance(len(p))
        nxt = int(logits[0, -1].argmax())
        n += 1
        pos = len(p)
        for _ in range(gen_len - 1):
            logits = model(torch.tensor([[nxt]], device=dev),
                           positions=torch.tensor([[pos]], device=dev), cache=cache)
            cache.advance(1)
            nxt = int(logits[0, -1].argmax())
            pos += 1
            n += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t0)


def bench_engine(model, tok, prompts: List[str], gen_len: int, batch: int,
                 prefix_cache: bool = False, kv_quant: bool = False,
                 cuda_graph: bool = False) -> tuple[float, Dict]:
    """L2+：PagedAttention + 连续批处理（可叠加 prefix cache / KV 量化 / CUDA Graph）。"""
    cuda = torch.cuda.is_available()
    cfg = EngineConfig(max_num_seqs=batch, block_size=16,
                       dtype="bf16" if cuda else "fp32",
                       device="cuda" if cuda else "cpu",
                       max_num_batched_tokens=batch * 256,
                       enable_prefix_caching=prefix_cache,
                       enable_cuda_graph=cuda_graph,
                       max_model_len=1024)
    engine = InferenceEngine(build_model(model.cfg), tok, cfg)
    engine.model.load_state_dict(model.state_dict())

    if kv_quant:
        h, d = model.cache_spec()
        q = QuantizedPagedKVCache(model.cfg.n_layers, engine.cache.num_blocks,
                                  cfg.block_size, h, d,
                                  dtype=engine.cache.k_cache.dtype,
                                  device=engine.cache.k_cache.device)
        q.allocator = engine.cache.allocator
        engine.cache = q
        engine.scheduler.set_allocator(q.allocator)

    params = SamplingParams(temperature=0.0, max_tokens=gen_len)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    outs = engine.generate(prompts, params)
    if cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0
    total = sum(o.completion_tokens for o in outs)
    return total / dt, engine.stats()


@torch.no_grad()
def bench_speculative(model, tok, prompts: List[str], gen_len: int,
                      device: torch.device, dtype: torch.dtype) -> float:
    """L6：投机解码（用同结构的浅层模型当 draft）。"""
    from tiniestgpt.inference.speculative import SpeculativeDecoder

    dev = torch.device(device)
    draft_cfg = ModelConfig(**{**vars(model.cfg), "n_layers": max(model.cfg.n_layers // 3, 1)})
    draft = build_model(draft_cfg).to(dev).to(dtype)
    sd = {k: v for k, v in model.state_dict().items() if k in draft.state_dict()}
    draft.load_state_dict(sd, strict=False)

    dec = SpeculativeDecoder(model, draft, tok, device=dev, dtype=dtype)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for p in prompts:
        r = dec.generate(tok.encode(p), SamplingParams(temperature=0.0, max_tokens=gen_len),
                         num_spec_tokens=4, max_new_tokens=gen_len, seed=0)
        n += len(r["output_ids"])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t0)


@torch.no_grad()
def bench_mtp_spec(model, tok, prompts: List[str], gen_len: int,
                   device: torch.device, dtype: torch.dtype):
    """L7：MTP 自草稿投机解码——**不需要独立的 draft 模型**。

    同时报告「平均接受长度 τ」：τ 必须 > 1 才有收益，
    否则多出来的验证前向反而更慢。这是判断投机解码是否值得的唯一硬指标。
    """
    from tiniestgpt.inference.mtp_spec import MTPSpecDecoder

    dev = torch.device(device)
    dec = MTPSpecDecoder(model, tok, device=dev, dtype=dtype)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n, rounds, accepted, bonus = 0, 0, 0, 0
    for p in prompts:
        r = dec.generate(tok.encode(p), SamplingParams(temperature=0.0, max_tokens=gen_len),
                         max_new_tokens=gen_len, seed=0)
        n += len(r["output_ids"])
        rounds += r["stats"].rounds
        accepted += r["stats"].accepted_tokens
        bonus += r["stats"].bonus_tokens
    if dev.type == "cuda":
        torch.cuda.synchronize()
    tps = n / (time.time() - t0)
    tau = (accepted + bonus) / max(rounds, 1)
    return tps, f"τ={tau:.2f} tok/轮（>1 才有收益）"


# --------------------------------------------------------------------------- #
def main() -> None:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser("inference ablation")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--preset", type=str, default="tiny")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.bfloat16 if cuda else torch.float32
    if args.checkpoint:
        model = load_model(args.checkpoint, map_location="cpu")
    else:
        model = build_model(args.preset)
    # 所有对比都在同一设备/同一 dtype 下进行，保证可比
    model = model.to(device).to(dtype).eval()

    tok = DummyTokenizer()
    if args.tokenizer:
        from tiniestgpt.data.tokenizer import Tokenizer

        tok = Tokenizer.load(args.tokenizer)

    prompts = [list(range(4, 4 + args.prompt_len)) for _ in range(args.batch)]
    prompt_texts = [tok.decode(p) for p in prompts]
    max_len = args.prompt_len + args.gen_len + 8

    rows = []

    def add(name: str, tps: float, note: str = "") -> None:
        rows.append((name, tps, note))

    add("L0 无 KV Cache（每步重算）", bench_no_cache(model, prompts[:2], 8, device))
    add("L1 稠密 KV Cache（单序列）",
        bench_dense_cache(model, prompts[:2], args.gen_len, max_len, device, dtype))

    tps, stats = bench_engine(model, tok, prompt_texts, args.gen_len, args.batch)
    add("L2 PagedAttention + 连续批处理", tps, f"batch={args.batch}")

    tps, stats = bench_engine(model, tok, prompt_texts, args.gen_len, args.batch,
                              prefix_cache=True)
    add("L3 + Prefix Caching", tps, f"命中率={stats.get('prefix_cache_hit_rate', 0):.0%}")

    try:
        tps, stats = bench_engine(model, tok, prompt_texts, args.gen_len, args.batch,
                                  kv_quant=True)
        add("L4 + KV Cache INT8 量化", tps)
    except Exception as exc:
        add("L4 + KV Cache INT8 量化", float("nan"), f"跳过: {exc}")

    if torch.cuda.is_available():
        tps, _ = bench_engine(model, tok, prompt_texts, args.gen_len, args.batch,
                              cuda_graph=True)
        add("L5 + CUDA Graph", tps)

    try:
        tps = bench_speculative(model, tok, prompt_texts[:2], args.gen_len, device, dtype)
        add("L6 投机解码（外挂 draft，γ=4）", tps)
    except Exception as exc:
        add("L6 投机解码（外挂 draft，γ=4）", float("nan"), f"跳过: {exc}")

    # L7：MTP 自草稿（不需要独立 draft 模型）——2026 主流做法
    if getattr(model.cfg, "mtp_enabled", False):
        try:
            tps, extra = bench_mtp_spec(model, tok, prompt_texts[:2], args.gen_len, device, dtype)
            add("L7 MTP 自草稿投机解码", tps, extra)
        except Exception as exc:
            add("L7 MTP 自草稿投机解码", float("nan"), f"跳过: {exc}")
    else:
        add("L7 MTP 自草稿投机解码", float("nan"),
            "跳过: 模型未启用 MTP（ModelConfig.mtp_enabled=True）")

    base = rows[0][1]
    print("\n" + "=" * 78)
    print(f"{'优化项':<34}{'吞吐 (tok/s)':>14}{'相对 L0':>10}   备注")
    print("-" * 78)
    for name, tps, note in rows:
        spd = f"{tps / base:.2f}x" if base == base and base > 0 else "-"
        print(f"{name:<34}{tps:>14,.1f}{spd:>10}   {note}")
    print("=" * 78)
    print(f"显存峰值: {_mem_mb():.0f} MB   设备: {next(model.parameters()).device}   dtype: {dtype}")
    print("怎么读这张表：")
    print("  * L1→L2 的跃升来自「批处理 + 分页显存」——把权重读取成本摊给多个请求；")
    print("  * L3 的收益与「前缀重复率」强相关，真实业务里系统提示词越长收益越大；")
    print("  * L4(KV INT8) 在本项目里走的是 PyTorch 参考实现（gather 后再反量化），")
    print("    额外开销会吃掉一部分收益；生产环境需要把 dequant 融进内核才能真正提速；")
    print("  * L6 的收益取决于 draft 与目标模型的**接受率**，小模型对之间往往不高，")
    print("    这也是为什么真实系统要用同族的小模型（或 EAGLE/Medusa 头）做 draft。\n")


if __name__ == "__main__":
    main()
