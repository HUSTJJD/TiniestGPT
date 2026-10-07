"""本项目引擎 vs vLLM 的横向对比。

目的不是"证明谁快"（vLLM 有完整的内核融合与并行，显然更快），
而是**把差距量化出来**：同样的模型、同样的负载，差多少倍、差在哪一层。

    uv run python scripts/export_hf.py --checkpoint out/tiny/last.pt --out-dir out/tiny_hf
    uv run python benchmarks/compare_vllm.py --model-dir out/tiny_hf \
        --checkpoint out/tiny/last.pt --tokenizer data/tokenizer.json

vLLM 未安装 / 非 Linux 时会打印安装指引并优雅退出（不会让仓库的测试失败）。
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import torch

from tiniestgpt.cli import _force_utf8_stdio


class DummyTokenizer:
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        return [4 + (i % 60) for i in range(min(len(text), 256))] or [4]

    def decode(self, ids, skip_special=True):
        return "".join(chr(40 + int(i) % 50) for i in ids)


def bench_ours(checkpoint: Optional[str], tokenizer, prompts: List[str],
               gen_len: int, batch: int, preset: str = "tiny") -> float:
    from tiniestgpt.inference.engine import EngineConfig, InferenceEngine
    from tiniestgpt.inference.sampler import SamplingParams
    from tiniestgpt.model.factory import build_model, load_model

    model = load_model(checkpoint, map_location="cpu") if checkpoint else build_model(preset)
    cuda = torch.cuda.is_available()
    cfg = EngineConfig(max_num_seqs=batch, block_size=16,
                       dtype="bf16" if cuda else "fp32",
                       device="cuda" if cuda else "cpu",
                       max_num_batched_tokens=batch * 256, max_model_len=1024)
    engine = InferenceEngine(model, tokenizer, cfg)
    params = SamplingParams(temperature=0.0, max_tokens=gen_len)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    outs = engine.generate(prompts, params)
    if cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0
    return sum(o.completion_tokens for o in outs) / dt


def bench_vllm(model_dir: str, prompts: List[str], gen_len: int) -> float:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError as exc:
        raise RuntimeError("vLLM 未安装") from exc

    llm = LLM(model=model_dir, gpu_memory_utilization=0.6, max_model_len=1024,
              enforce_eager=False)
    params = SamplingParams(temperature=0.0, max_tokens=gen_len)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outputs)
    return n / dt


def main() -> None:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser("compare with vLLM")
    ap.add_argument("--model-dir", type=str, required=True, help="export_hf.py 导出的目录")
    ap.add_argument("--checkpoint", type=str, default=None, help="本项目 checkpoint（用于跑我们的引擎）")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--preset", type=str, default="tiny")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=32)
    args = ap.parse_args()

    tok = DummyTokenizer()
    if args.tokenizer:
        from tiniestgpt.data.tokenizer import Tokenizer

        tok = Tokenizer.load(args.tokenizer)

    prompts = [tok.decode(list(range(4, 4 + args.prompt_len))) for _ in range(args.batch)]

    print("=" * 72)
    print(f"{'引擎':<28}{'吞吐 (tok/s)':>14}{'相对':>10}   备注")
    print("-" * 72)

    ours = bench_ours(args.checkpoint, tok, prompts, args.gen_len, args.batch, args.preset)
    print(f"{'TiniestGPT 引擎':<28}{ours:>14,.1f}{'1.00x':>10}   PyTorch 参考内核")

    try:
        v = bench_vllm(args.model_dir, prompts, args.gen_len)
        print(f"{'vLLM（同一份权重）':<28}{v:>14,.1f}{v / max(ours, 1e-9):>9.2f}x   "
              f"内核融合 + 生产调度")
        print("-" * 72)
        print("差距来源（按量级从大到小）：")
        print("  1. 内核：vLLM 的 PagedAttention / dequant+GEMM 是融合内核；")
        print("     本项目是 PyTorch 参考实现，中间张量与 launch 开销都高。")
        print("  2. 批处理：vLLM 的 prefill 与 decode 混在同一 batch；本项目分两次前向。")
        print("  3. 图捕获：vLLM 的 CUDA Graph 覆盖更全、分桶更细。")
        print("  4. 调度：vLLM 是多进程 + 异步输出处理，CPU 侧不阻塞。")
    except Exception as exc:
        print(f"{'vLLM':<28}{'-':>14}{'-':>10}   跳过: {exc}")
        print("-" * 72)
        print("vLLM 仅支持 Linux，且会约束 numpy 等基础依赖版本，因此刻意不放进 extras。")
        print("建议在独立环境里安装（不污染主环境）：")
        print("    uv venv --python 3.12 .venv-vllm")
        print("    uv pip install --python .venv-vllm vllm")
        print("    .venv-vllm/Scripts/python benchmarks/compare_vllm.py --model-dir out/tiny_hf")
    print("=" * 72)


if __name__ == "__main__":
    main()
