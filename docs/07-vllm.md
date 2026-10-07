# vLLM 对照：从"教学实现"到"生产实现"

vLLM 是当前事实上的开源推理引擎标准。本项目**刻意重新实现**了它的核心机制
（而不是直接调库），目的就是让每个机制的原理可见；但生产实现里有大量
本项目为了可读性而**刻意省略**的工程细节。本文就是两者的对照表。

## 1. 模块对照表

| 本项目 | vLLM | 职责 |
|---|---|---|
| `model/kv_cache.py::PagedKVCache` | `vllm/v1/worker/kv_cache_utils.py` + `block_pool.py` | 分页 KV 的物理存储 |
| `model/kv_cache.py::BlockAllocator` | `vllm/v1/core/block_pool.py::BlockPool` | 块分配 / 引用计数 / 释放 |
| `inference/scheduler.py::Scheduler` | `vllm/v1/core/sched/scheduler.py` | 每步调度：谁 prefill、谁 decode |
| `inference/scheduler.py::Sequence` | `vllm/sequence.py::SequenceGroup` | 请求状态与元数据 |
| `inference/scheduler.py::PrefixCache` | `vllm/v1/core/kv_cache_coordinator.py`（prefix cache 相关） | 按块哈希复用 KV |
| `inference/engine.py::InferenceEngine` | `vllm/v1/engine/core.py` + `entrypoints/llm.py` | 主循环与对外 API |
| `model/kernels.py::paged_attention_ref` | `vllm/attention/ops/paged_attn.py` | PagedAttention 的 PyTorch 参考实现 |
| `inference/kernels/triton_kernels.py` | `vllm/attention/ops/paged_attn.py`（Triton 版）、`vllm/attention/backends/flash_attn.py` | 高性能内核 |
| `inference/sampler.py` | `vllm/v1/sample/sampler.py` + `ops/{topk_topp_sampler,penalties}.py` | 采样与 logits 处理 |
| `inference/quantization/` | `vllm/model_executor/layers/quantization/` | GPTQ / AWQ / SmoothQuant / FP8 |
| `inference/speculative.py` | `vllm/v1/spec_decode/` | 投机解码（draft / EAGLE / Medusa / ngram） |
| `inference/cuda_graph.py` | `vllm/v1/worker/gpu_model_runner.py` 中的 `CUDAGraphRunner` | CUDA Graph 捕获与分桶 |
| `inference/server.py` | `vllm/entrypoints/openai/api_server.py` | OpenAI 兼容 HTTP 服务 |

> vLLM 的 v0/v1 目录差异很大（v1 自 0.9 起成为默认）。上表以 **v1** 为主。

## 2. vLLM 做了、而本项目为了可读性省略的

理解这张表，就知道"从教学实现到生产实现"还差多远：

| 机制 | vLLM | 本项目 |
|---|---|---|
| **张量并行 / 流水线并行** | 完整支持（`vllm/distributed/`，含 Megatron 风格的 column/row parallel linear） | 未实现（单卡） |
| **Chunked prefill 的调度细节** | 长 prompt 与 decode 混在同一 batch，按 token 预算切分 | 已实现预算切分，但 prefill 与 decode **分两次前向** |
| **注意力后端矩阵** | FlashAttention / FlashInfer / Triton / FlexAttention / ROCm，按硬件自动选 | SDPA + 自写 Triton decode 内核 |
| **CUDA Graph 覆盖范围** | 覆盖 prefill 与 decode，按 batch 分桶，含 attention 的 persistent buffer | 仅 decode，按 (batch, 长度档位) 分桶 |
| **前缀缓存淘汰** | LRU，按块引用计数与哈希链管理 | 有哈希复用，无淘汰策略 |
| **抢占策略** | recompute + swap（CPU 交换空间）双策略 | 仅 recompute |
| **多 LoRA / 多模型** | 支持 | 不支持 |
| **结构化输出 / 语法约束解码** | `vllm/v1/structured_output/`，支持 JSON Schema / 正则 | 未实现（采样层留了接口） |
| **指标与可观测** | Prometheus 全套（TTFT / TPOT / 排队长度 / 缓存命中率） | `engine.stats()` 基础指标 |
| **sleep / wake、权重在线更新** | 支持 | 不支持 |
| **量化内核** | Marlin / Machete / CUTLASS，dequant 与 GEMM 融合 | dequant 与 GEMM 分离（参考实现） |

## 3. 本项目比 vLLM"更适合学习"的地方

- **每个内核都有等价的 PyTorch 参考实现**，可以做数值比对；vLLM 里只有优化版。
- **调度器是纯 Python、无异步无进程间通信**，可以单步调试；vLLM v1 是
  多进程（EngineCore 在独立进程）+ ZeroMQ，阅读成本高。
- **模型与引擎解耦**：`Transformer` 不持有 KV Cache 状态，
  所以同一份权重能同时跑训练、稠密缓存推理、分页推理。

## 4. 把本项目模型交给 vLLM 跑

我们的架构（RMSNorm + SwiGLU + GQA + RoPE）与 LLaMA 同构，
因此可以导出成 HF `LlamaForCausalLM` 格式，直接喂给 vLLM：

```bash
# 1) 导出（config.json + 权重，LLaMA 兼容命名）
uv run python scripts/export_hf.py --checkpoint out/tiny/last.pt \
    --tokenizer data/tokenizer.json --out-dir out/tiny_hf

# 2) 用 vLLM 加载（需要 Linux + vllm）
uv sync --extra all --extra vllm        # 仅 Linux
uv run python -m vllm.entrypoints.openai.api_server --model out/tiny_hf

# 3) 与本项目引擎横向对比
uv run python benchmarks/compare_vllm.py --model-dir out/tiny_hf \
    --checkpoint out/tiny/last.pt --tokenizer data/tokenizer.json
```

> ⚠️ 注意：导出的模型在 vLLM 下**只是跑得快**，并不代表效果更好——
> 分词器是我们自己训的小 BPE，权重也只在玩具语料上训过。
> 这个流程的价值在于**打通"自己训的模型 → 生产引擎"这条链路**。

## 5. 阅读 vLLM 源码的建议顺序

配合本项目的同名模块一起读，效率最高：

1. `vllm/v1/core/sched/scheduler.py` ← 先读 `inference/scheduler.py`
2. `vllm/v1/core/block_pool.py` ← 先读 `model/kv_cache.py::BlockAllocator`
3. `vllm/v1/worker/gpu_model_runner.py` ← 先读 `inference/engine.py` + `cuda_graph.py`
4. `vllm/attention/ops/paged_attn.py` ← 先读 `model/kernels.py::paged_attention_ref`
5. `vllm/model_executor/layers/quantization/*` ← 先读 `inference/quantization/`

每次读之前先问自己："如果让我实现，我会怎么写？"——本项目的代码就是这个问题的答案，
两者一对照，差异点就是生产系统的真正价值所在。
