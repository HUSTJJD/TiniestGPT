"""统一 CLI 入口：``python -m tiniestgpt.cli <子命令>``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]

_SUBCOMMANDS = ("data", "pretrain", "generate", "serve", "agent", "quantize", "bench", "info")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", nargs="*", default=[], help="点号覆盖，如 model.n_layers=8")


def _force_utf8_stdio() -> None:
    """Windows 中文控制台默认用 GBK，遇到 U+FFFD 等字符会抛 UnicodeEncodeError。

    这里把标准流强制改成 UTF-8 + errors="replace"，避免"模型生成出乱码却把程序搞崩"。
    """
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def main(argv=None) -> int:
    _force_utf8_stdio()
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print("用法: python -m tiniestgpt.cli {" + ",".join(_SUBCOMMANDS) + "} [options]")
        return 0
    cmd, rest = argv[0], argv[1:]

    # ---------------------------------------------------------------- data
    if cmd == "data":
        from .common.config import from_dict, load_config
        from .data.pipeline import DataPipelineConfig, run_pipeline

        p = argparse.ArgumentParser("data")
        p.add_argument("--config", type=str, default=None)
        p.add_argument("--out-dir", type=str, default=None)
        p.add_argument("--source", type=str, default=None)      # toy | jsonl
        p.add_argument("--jsonl", type=str, default=None)
        p.add_argument("--vocab-size", type=int, default=None)
        p.add_argument("--n-docs", type=int, default=None)
        _add_common(p)
        a = p.parse_args(rest)
        cfg = (load_config(DataPipelineConfig, a.config, a.set)
               if a.config else from_dict(DataPipelineConfig, {}))
        if a.out_dir:
            cfg.out_dir = a.out_dir
        if a.source:
            cfg.source = a.source
        if a.jsonl:
            cfg.source, cfg.source_path = "jsonl", a.jsonl
        if a.vocab_size:
            cfg.vocab_size = a.vocab_size
        if a.n_docs:
            cfg.n_docs = a.n_docs
        run_pipeline(cfg)
        return 0

    # ------------------------------------------------------------ pretrain
    if cmd == "pretrain":
        from .train.pretrain import main as _pretrain

        _pretrain(rest)
        return 0

    # ------------------------------------------------------------ generate
    if cmd == "generate":
        p = argparse.ArgumentParser("generate")
        p.add_argument("--checkpoint", type=str, required=True)
        p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
        p.add_argument("--prompt", type=str, default="Once upon a time")
        p.add_argument("--prompts-file", type=str, default=None, help="每行一个 prompt")
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--top-k", type=int, default=0)
        p.add_argument("--min-p", type=float, default=0.0)
        p.add_argument("--num-seqs", type=int, default=8)
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--dtype", type=str, default="bf16")
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--no-prefix-cache", action="store_true")
        p.add_argument("--cuda-graph", action="store_true")
        p.add_argument("--preset", type=str, default=None)
        a = p.parse_args(rest)

        from .data.tokenizer import Tokenizer
        from .inference.engine import EngineConfig, InferenceEngine
        from .inference.sampler import SamplingParams
        from .model.factory import build_model, load_model

        tok = Tokenizer.load(a.tokenizer)
        model = load_model(a.checkpoint, map_location="cpu") if Path(a.checkpoint).exists() \
            else build_model(a.preset or "tiny")
        ecfg = EngineConfig(max_num_seqs=a.num_seqs, block_size=a.block_size, dtype=a.dtype,
                            device=a.device, enable_prefix_caching=not a.no_prefix_cache,
                            enable_cuda_graph=a.cuda_graph)
        engine = InferenceEngine(model, tok, ecfg)
        prompts = ([l for l in Path(a.prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
                   if a.prompts_file else [a.prompt])
        params = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                min_p=a.min_p, max_tokens=a.max_tokens)
        outs = engine.generate(prompts, params, verbose=True)
        for o in outs:
            print(f"\n=== #{o.seq_id} ({o.completion_tokens} tokens) ===\n{o.text}")
        print("\n[engine stats]", json.dumps(engine.stats(), ensure_ascii=False, indent=2))
        return 0

    # --------------------------------------------------------------- serve
    if cmd == "serve":
        p = argparse.ArgumentParser("serve")
        p.add_argument("--checkpoint", type=str, required=True)
        p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
        p.add_argument("--host", type=str, default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--dtype", type=str, default="bf16")
        a = p.parse_args(rest)

        from .data.tokenizer import Tokenizer
        from .inference.engine import EngineConfig, InferenceEngine
        from .inference.server import run_server
        from .model.factory import load_model

        model = load_model(a.checkpoint, map_location="cpu")
        tok = Tokenizer.load(a.tokenizer)
        engine = InferenceEngine(model, tok, EngineConfig(dtype=a.dtype))
        run_server(engine, host=a.host, port=a.port)
        return 0

    # --------------------------------------------------------------- agent
    if cmd == "agent":
        p = argparse.ArgumentParser("agent")
        p.add_argument("--backend", type=str, default="echo")     # echo | local | openai
        p.add_argument("--task", type=str, required=True)
        p.add_argument("--checkpoint", type=str, default=None)
        p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
        p.add_argument("--planner", type=str, default="react")
        p.add_argument("--max-steps", type=int, default=8)
        p.add_argument("--allow-python", action="store_true")
        p.add_argument("--trace", type=str, default=None)
        p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
        a = p.parse_args(rest)

        from .agent import Agent, AgentConfig, EchoBackend, LocalEngineBackend, OpenAIBackend
        from .agent.builtin_tools import build_default_registry

        if a.backend == "echo":
            backend = EchoBackend(replies=[
                'Thought: 需要调用计算器\nAction: calculator\n'
                'Action Input: {"expression": "' + a.task.replace('"', '') + '"}',
                "Final Answer: 已使用工具完成任务。",
            ])
        elif a.backend == "local":
            from .inference.engine import EngineConfig, InferenceEngine
            from .model.factory import load_model

            assert a.checkpoint, "--backend local 需要 --checkpoint"
            from .data.tokenizer import Tokenizer

            engine = InferenceEngine(load_model(a.checkpoint, map_location="cpu"),
                                     Tokenizer.load(a.tokenizer), EngineConfig())
            backend = LocalEngineBackend(engine)
        else:
            backend = OpenAIBackend(base_url=a.base_url)

        reg = build_default_registry(allow_python=a.allow_python)
        cfg = AgentConfig(planner=a.planner, max_steps=a.max_steps, trace_path=a.trace)
        agent = Agent(backend, reg, cfg=cfg)
        result = agent.run(a.task)
        print("\n=== ANSWER ===\n" + result.answer)
        print("\n" + result.summary())
        return 0

    # ------------------------------------------------------------ quantize
    if cmd == "quantize":
        p = argparse.ArgumentParser("quantize")
        p.add_argument("--checkpoint", type=str, required=True)
        p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
        p.add_argument("--method", type=str, default="rtn")     # rtn | gptq
        p.add_argument("--bits", type=int, default=4)
        p.add_argument("--group-size", type=int, default=128)
        p.add_argument("--out", type=str, default=None)
        p.add_argument("--eval-prompts", type=int, default=0)
        a = p.parse_args(rest)

        import torch

        from .data.tokenizer import Tokenizer
        from .model.factory import load_model
        from .inference.quantization import quantize_model, gptq_quantize_model
        from .inference.quantization.calibration import CalibrationCollector

        tok = Tokenizer.load(a.tokenizer)
        model = load_model(a.checkpoint, map_location="cpu")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        if a.method == "gptq":
            # 用随机 token 序列做校准（真实场景请用训练语料）
            batches = [torch.randint(0, model.cfg.vocab_size, (2, 128)) for _ in range(8)]
            with CalibrationCollector(model, max_samples_per_module=1024) as c:
                calib = c.collect(batches, max_batches=8)
            model = gptq_quantize_model(model, calib, bits=a.bits, group_size=a.group_size)
        else:
            model = quantize_model(model, bits=a.bits, group_size=a.group_size)

        out = a.out or a.checkpoint.replace(".pt", f"_int{a.bits}.pt")
        torch.save({"model": model.state_dict(), "config": vars(model.cfg)}, out)
        print(f"[quantize] saved -> {out}")
        return 0

    # ---------------------------------------------------------------- bench
    if cmd == "bench":
        import runpy

        from pathlib import Path as _P

        script = _P(__file__).resolve().parent.parent / "benchmarks" / "inference_ablation.py"
        sys.argv = [str(script)] + rest
        runpy.run_path(str(script), run_name="__main__")
        return 0

    # ---------------------------------------------------------------- info
    if cmd == "info":
        from .model.config import PRESETS
        from .model.factory import build_model
        from .model.dispatch import available_backends

        for name in PRESETS:
            m = build_model(name)
            print(m.summary())
            print()
        print("[kernels]", available_backends())
        return 0

    print(f"未知子命令: {cmd}；可用: {_SUBCOMMANDS}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
