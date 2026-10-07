"""推理引擎：把 PagedAttention + 连续批处理 + Prefix Caching 组装成一个系统。

一次 ``step()`` 的完整流程::

    schedule()            → 决定本步要 prefill 谁、要 decode 谁（受 token 预算与显存块限制）
      ├─ _run_prefill()   → 分块预填充（可只算未缓存前缀），写 KV，最后一块才采样
      └─ _run_decode()    → 所有 running 序列各喂 1 个 token（batch=B），写 KV，采样
    append / finish       → 完成的序列立刻释放块，等待队列立刻补位

关键设计：
  * **模型不持有状态**：KV Cache 由引擎分配并通过参数注入，
    因此同一份权重可以被多个引擎 / 多种批处理策略复用；
  * **prefill 与 decode 分开前向**：形状规则（decode 全是 T=1），
    便于后续接 CUDA Graph 与张量并行；
  * **每一步都是一次完整的调度决策**，这就是 continuous batching 的本质。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..common.logging import get_logger
from ..common.profiler import AvgMeter, Timer
from ..model.factory import load_model
from ..model.kv_cache import PagedKVCache
from .sampler import SamplingParams, Sampler
from .scheduler import Scheduler, Sequence, SequenceStatus

log = get_logger("tiniestgpt.engine")

__all__ = ["EngineConfig", "RequestOutput", "InferenceEngine"]


@dataclass
class EngineConfig:
    max_model_len: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048     # 每步 token 预算（决定 prefill 分块粒度）
    block_size: int = 16
    num_gpu_blocks: Optional[int] = None   # None → 按显存自动推断
    gpu_memory_utilization: float = 0.85
    dtype: str = "bf16"                    # fp32 | bf16 | fp16
    device: str = "auto"
    kv_cache_dtype: Optional[str] = None   # None = 同 dtype；可选 fp8 / int8
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_cuda_graph: bool = False
    cuda_graph_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    swap_space_blocks: int = 0             # >0 时启用 CPU 交换空间（抢占走 swap 而非重算）
    prefix_cache_impl: str = "hash"        # hash（默认） | radix（前缀树 + LRU 淘汰）
    preemption_mode: str = "recompute"     # recompute | swap（需 swap_space_blocks>0）


@dataclass
class RequestOutput:
    seq_id: int
    output_ids: List[int]
    text: str = ""
    finished: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def __repr__(self) -> str:
        return (f"RequestOutput(id={self.seq_id}, finished={self.finished}, "
                f"tokens={self.completion_tokens}, {self.latency_ms:.0f}ms)")


class InferenceEngine:
    def __init__(self, model, tokenizer, cfg: Optional[EngineConfig] = None) -> None:
        self.cfg = cfg or EngineConfig()
        self.tokenizer = tokenizer
        self.model = model
        self.device = self._resolve_device(self.cfg.device)
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                      "fp16": torch.float16}[self.cfg.dtype]
        if self.dtype != torch.float32:
            self.model = self.model.to(self.dtype)
        self.model = self.model.to(self.device).eval()
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

        # ---------------- 分页 KV Cache ----------------
        n_kv_heads, head_dim = self.raw_model.cache_spec()
        self.kv_dtype = self._resolve_kv_dtype()
        num_blocks = self.cfg.num_gpu_blocks or self._infer_num_blocks()
        self.cache = PagedKVCache(
            n_layers=self.raw_model.cfg.n_layers, num_blocks=num_blocks,
            block_size=self.cfg.block_size, n_kv_heads=n_kv_heads, head_dim=head_dim,
            dtype=self.kv_dtype, device=self.device,
        )
        log.info("KV cache: %d blocks × %d tokens = %d tokens (%.1f MB, dtype=%s)",
                 num_blocks, self.cfg.block_size, num_blocks * self.cfg.block_size,
                 self.cache.memory_bytes / 1024 ** 2, self.kv_dtype)

        # ---------------- 调度器 ----------------
        self.scheduler = Scheduler(
            block_size=self.cfg.block_size, max_num_seqs=self.cfg.max_num_seqs,
            max_num_batched_tokens=self.cfg.max_num_batched_tokens,
            enable_chunked_prefill=self.cfg.enable_chunked_prefill,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            max_model_len=self.cfg.max_model_len,
            prefix_cache_impl=self.cfg.prefix_cache_impl,
            preemption_mode=self.cfg.preemption_mode,
        )
        self.scheduler.set_allocator(self.cache.allocator)

        # ---------------- CPU 交换空间（可选） ----------------
        self.swap_space = None
        if self.cfg.swap_space_blocks > 0:
            from .swap import CPUSwapSpace

            self.swap_space = CPUSwapSpace(
                self.cache.k_cache, self.cache.v_cache, self.cfg.swap_space_blocks)
            self.scheduler.set_swap_space(self.swap_space)
            log.info("CPU swap space: %d blocks (%.1f MB)", self.cfg.swap_space_blocks,
                     self.swap_space.k.numel() * self.swap_space.k.element_size() * 2 / 1024 ** 2)

        self.sampler = Sampler(self.device)
        self.seq_counter = 0
        self.outputs: Dict[int, RequestOutput] = {}
        self.step_timer = Timer(sync_cuda=self.device.type == "cuda")
        self.decode_meter = AvgMeter(window=100)
        self.prefill_meter = AvgMeter(window=100)

        # ---------------- CUDA Graph（可选） ----------------
        self.cuda_graph = None
        if self.cfg.enable_cuda_graph and self.device.type == "cuda":
            try:
                from .cuda_graph import CUDAGraphManager

                self.cuda_graph = CUDAGraphManager(
                    self.raw_model, self.cache, self.cfg.cuda_graph_batch_sizes,
                    max_blocks=self.cfg.max_model_len // self.cfg.block_size + 1)
            except Exception as exc:
                log.warning("CUDA Graph 不可用，回退到 eager: %s", exc)

    # ------------------------------------------------------------------ #
    def _resolve_device(self, spec: str) -> torch.device:
        if spec != "auto":
            return torch.device(spec)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_kv_dtype(self) -> torch.dtype:
        if self.cfg.kv_cache_dtype in ("fp8",):
            return torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16
        return self.dtype

    def _infer_num_blocks(self) -> int:
        """按剩余显存推断可分配的 KV 块数（vLLM 的 ``gpu_memory_utilization`` 思路）。"""
        if self.device.type != "cuda":
            return 4096
        total, _ = torch.cuda.mem_get_info()
        used = torch.cuda.memory_allocated()
        budget = int(total * self.cfg.gpu_memory_utilization) - used
        per_token = self.raw_model.kv_cache_bytes_per_token(self.kv_dtype)
        per_block = per_token * self.cfg.block_size
        n = max(int(budget // max(per_block, 1)), 16)
        return min(n, 65536)

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def add_request(self, prompt: str | Sequence[int],
                    sampling_params: Optional[SamplingParams] = None,
                    json_schema: Optional[dict] = None) -> int:
        """:param json_schema: 给定时开启**约束解码**，保证输出是合法 JSON（见 structured.py）。"""
        ids = prompt if not isinstance(prompt, str) else self.tokenizer.encode(prompt)
        params = sampling_params or SamplingParams()
        params.validate()
        self.seq_counter += 1
        seq = Sequence(seq_id=self.seq_counter, prompt_ids=list(ids),
                       sampling_params=params, block_size=self.cfg.block_size)
        if json_schema is not None or getattr(params, "json_schema", None) is not None:
            schema = json_schema if json_schema is not None else params.json_schema
            from .structured import StructuredDecoder

            seq.structured = StructuredDecoder(      # type: ignore[attr-defined]
                self.tokenizer, self.model.cfg.vocab_size, schema)
        self.scheduler.add_seq(seq)
        self.outputs[seq.seq_id] = RequestOutput(
            seq_id=seq.seq_id, output_ids=[], prompt_tokens=seq.prompt_len,
            latency_ms=0.0)
        return seq.seq_id

    # ------------------------------------------------------------------ #
    def _sample_token(self, seq: Sequence, logits: torch.Tensor) -> int:
        """采样一个 token：先做约束解码的 logits masking，再交给 Sampler。"""
        dec = getattr(seq, "structured", None)
        if dec is not None:
            logits = dec.mask_logits(logits)
        return self.sampler.sample(logits, seq.sampling_params,
                                   prompt_ids=seq.prompt_ids, output_ids=seq.output_ids)

    def _after_token(self, seq: Sequence, token_id: int) -> None:
        """采样后：推进约束状态机，JSON 闭合即可提前结束（不用等 max_tokens）。"""
        dec = getattr(seq, "structured", None)
        if dec is None:
            return
        dec.feed_token(token_id)
        if dec.complete:
            seq._force_finish = True            # type: ignore[attr-defined]

    def step(self) -> List[RequestOutput]:
        """执行一步调度 + 前向 + 采样。返回本步有更新的请求。"""
        sched_out = self.scheduler.schedule()
        updated: List[RequestOutput] = []
        if not len(sched_out):
            return updated

        self.step_timer.start()
        if sched_out.prefill_seqs:
            self._run_prefill(sched_out.prefill_seqs)
        if sched_out.decode_seqs:
            self._run_decode(sched_out.decode_seqs)
        self.step_timer.stop()

        for seq in list(self.scheduler.running):
            out = self.outputs[seq.seq_id]
            out.output_ids = list(seq.output_ids)
            out.completion_tokens = seq.output_len
            if seq.is_finished() or seq.get_len() >= self.cfg.max_model_len:
                out.text = self.tokenizer.decode(seq.output_ids)
                out.finished = True
                self.scheduler.finish_seq(seq)
            updated.append(out)
        return updated

    def generate(self, prompts: Sequence[str] | str,
                 sampling_params: Optional[SamplingParams] = None,
                 verbose: bool = False) -> List[RequestOutput]:
        """同步生成（内部驱动 ``step()`` 直到全部完成）。"""
        if isinstance(prompts, str):
            prompts = [prompts]
        ids = []
        for p in prompts:
            ids.append(p if not isinstance(p, str) else self.tokenizer.encode(p))
        t0 = time.time()
        for x in ids:
            self.add_request(x, sampling_params)
        results: Dict[int, RequestOutput] = {}
        while self.scheduler.has_unfinished():
            for out in self.step():
                results[out.seq_id] = out
                if verbose and out.finished:
                    log.info("  [done] #%d %d tokens | %s", out.seq_id,
                             out.completion_tokens, out.text[:80].replace("\n", " "))
        elapsed = (time.time() - t0) * 1000
        outs = []
        for sid in sorted(results):
            o = results[sid]
            o.latency_ms = elapsed
            o.text = self.tokenizer.decode(o.output_ids)
            outs.append(o)
        return outs

    # ------------------------------------------------------------------ #
    # prefill / decode
    # ------------------------------------------------------------------ #
    def _slot_ids(self, seq: Sequence, start: int, length: int) -> List[int]:
        bs = self.cfg.block_size
        out = []
        for i in range(start, start + length):
            b = seq.block_table[i // bs]
            out.append(b * bs + i % bs)
        return out

    @torch.no_grad()
    def _run_prefill(self, seqs: List[Sequence]) -> None:
        B = len(seqs)
        chunks = [int(getattr(s, "_chunk", s.prompt_len - s.computed)) for s in seqs]
        max_len = max(chunks)
        pad_id = getattr(self.tokenizer, "pad_id", 0)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        positions = torch.zeros((B, max_len), dtype=torch.long, device=self.device)
        slot_ids = torch.full((B, max_len), -1, dtype=torch.long, device=self.device)
        seq_lens = torch.zeros(B, dtype=torch.long, device=self.device)
        max_blocks = max(len(s.block_table) for s in seqs)
        block_table = torch.full((B, max_blocks), -1, dtype=torch.long, device=self.device)

        for i, (s, c) in enumerate(zip(seqs, chunks)):
            start = s.computed
            toks = s.prompt_ids[start:start + c]
            input_ids[i, :len(toks)] = torch.tensor(toks, device=self.device)
            positions[i, :len(toks)] = torch.arange(start, start + len(toks), device=self.device)
            sl = self._slot_ids(s, start, c)
            slot_ids[i, :len(sl)] = torch.tensor(sl, device=self.device)
            seq_lens[i] = start + c
            if s.block_table:
                block_table[i, :len(s.block_table)] = torch.tensor(s.block_table, device=self.device)

        self.cache.set_batch(block_table=block_table, seq_lens=seq_lens, slot_ids=slot_ids)
        logits = self.model(input_ids, positions=positions, cache=self.cache)

        # 只有"本次把 prompt 算完"的序列才采样
        new_tokens: Dict[int, int] = {}
        for i, (s, c) in enumerate(zip(seqs, chunks)):
            s.computed += c
            if s.computed >= s.prompt_len and s.output_len == 0:
                last = logits[i, c - 1].float()
                tok = self._sample_token(s, last)
                new_tokens[s.seq_id] = tok
        for s in seqs:
            if s.seq_id in new_tokens:
                s.append_token(new_tokens[s.seq_id])
                self._after_token(s, new_tokens[s.seq_id])
        self.prefill_meter.update(sum(chunks))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _run_decode(self, seqs: List[Sequence]) -> None:
        B = len(seqs)
        max_blocks = max(len(s.block_table) for s in seqs)

        input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        positions = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        slot_ids = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        seq_lens = torch.zeros(B, dtype=torch.long, device=self.device)
        block_table = torch.full((B, max_blocks), -1, dtype=torch.long, device=self.device)

        for i, s in enumerate(seqs):
            last_tok = s.output_ids[-1] if s.output_ids else s.prompt_ids[-1]
            pos = s.get_len() - 1
            input_ids[i, 0] = last_tok
            positions[i, 0] = pos
            slot_ids[i, 0] = self._slot_ids(s, pos, 1)[0]
            seq_lens[i] = pos + 1
            if s.block_table:
                block_table[i, :len(s.block_table)] = torch.tensor(s.block_table, device=self.device)

        logits = None
        if self.cuda_graph is not None:
            # 命中 (batch_size, 长度档位) 的图就 replay；否则退回 eager
            logits = self.cuda_graph.forward(input_ids, positions, block_table, slot_ids, seq_lens)
        if logits is None:
            self.cache.set_batch(block_table=block_table, seq_lens=seq_lens, slot_ids=slot_ids)
            logits = self.model(input_ids, positions=positions, cache=self.cache)

        for i, s in enumerate(seqs):
            tok = self._sample_token(s, logits[i, 0].float())
            s.append_token(tok)
            self._after_token(s, tok)
        self.decode_meter.update(B)

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, float]:
        total_tokens = sum(o.completion_tokens for o in self.outputs.values())
        return {
            "steps": self.scheduler.stats["steps"],
            "preempted": self.scheduler.stats["preempted"],
            "block_usage": self.cache.allocator.usage,
            "free_blocks": self.cache.allocator.num_free,
            "prefix_cache_hit_rate": self.scheduler.prefix_cache.hit_rate,
            "total_generated_tokens": total_tokens,
            "avg_step_ms": self.step_timer.mean * 1000,
            "avg_decode_batch": self.decode_meter.avg,
            "kv_cache_mb": self.cache.memory_bytes / 1024 ** 2,
        }

    def reset(self) -> None:
        self.scheduler.abort_all()
        self.cache.reset()
        self.scheduler.set_allocator(self.cache.allocator)
        self.outputs.clear()


def build_engine(checkpoint: str, tokenizer=None, cfg: Optional[EngineConfig] = None,
                 model=None) -> InferenceEngine:
    """从 checkpoint 一步构建引擎。"""
    if model is None:
        model = load_model(checkpoint, map_location="cpu")
    if tokenizer is None:
        from ..data.tokenizer import Tokenizer

        tok_path = None
        import os

        for cand in (os.path.join(os.path.dirname(checkpoint), "..", "data", "tokenizer.json"),
                     "data/tokenizer.json"):
            if os.path.exists(cand):
                tok_path = cand
                break
        if tok_path is None:
            raise RuntimeError("找不到分词器，请显式传入 tokenizer")
        tokenizer = Tokenizer.load(tok_path)
    return InferenceEngine(model, tokenizer, cfg)
