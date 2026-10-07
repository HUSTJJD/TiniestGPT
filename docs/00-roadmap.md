# 学习路线：如何用 TiniestGPT 学完大模型全链路

## 0. 这个项目的定位

大模型工程可以拆成六个层次，每一层的"核心矛盾"都不同：

| 层 | 核心矛盾 | 关键指标 | 本项目对应 |
|---|---|---|---|
| 数据 | 质量 vs 规模 | 去重率、质量分分布、B/token | `tiniestgpt/data/` |
| 架构 | 表达能力 vs 计算/显存 | loss/参数、KV/token | `tiniestgpt/model/` |
| 训练 | 收敛速度 vs 稳定性 vs 成本 | tokens/s、MFU、loss 尖刺 | `tiniestgpt/train/` |
| 后训练 | 能力 vs 对齐 | 偏好准确率、GRPO 奖励 | `tiniestgpt/posttrain/` |
| 推理 | 延迟 vs 吞吐 vs 显存 | tok/s、p99、显存占用 | `tiniestgpt/inference/` |
| 应用 | 自主性 vs 可控性 | 任务成功率、成本 | `tiniestgpt/agent/` |

**"小而全"的关键**：把参数量压到 25M，任何一个实验都能在几分钟内跑完，
于是你可以把精力放在"理解为什么"而不是"等待结果"。

## 1. 建议的学习顺序

### 阶段 0：跑通（半天）
```bash
# 环境（uv 管理，CUDA 版 torch）
.\scripts\setup.ps1                # Linux/macOS: bash scripts/setup.sh

# 全链路
uv run python -m tiniestgpt.cli data      --config recipes/data_toy.yaml
uv run python -m tiniestgpt.cli pretrain  --config recipes/pretrain_tiny.yaml
uv run python -m tiniestgpt.cli generate  --checkpoint out/tiny/last.pt --prompt "Once upon a time"
```
先看懂 `data/pipeline_report.json`：每个阶段过滤掉了多少、为什么。

### 阶段 1：数据（1~2 天）
读 `01-data.md`。动手实验：
- 把 `max_ngram_dup_ratio` 从 0.30 调到 0.05，看过滤率与 BPE 压缩率怎么变；
- 打开 `train_classifier=true`，比较启发式与判别器的打分分布；
- 对比三种打包策略的 `pad_ratio`（`naive` vs `concat` vs `bfd`）。

### 阶段 2：架构（2~3 天）
读 `02-architecture.md`。逐个开关 `ModelConfig` 的字段，**每次都跑一次 `tests/test_model.py`**：
- `attn_type: gqa → mha → mla`，观察 `uv run python -m tiniestgpt.cli info` 里的 `kv/token`；
- `attn_window: -1 → 256`，配合 `attn_sinks: 0 vs 4`，理解"窗口必须配 sink"；
- `moe_enabled: true`，读日志里的 `aux` 与 MoE 负载统计；
- `layer_types: "window,linear,window,full"`，理解混合架构。

### 阶段 3：训练（2 天）
读 `03-training.md`。
- 对比 `optimizer: muon vs adamw` 在相同步数下的 loss；
- 把 `lr_schedule: wsd → cosine`，然后在中途调用 `scheduler.start_decay()` 体会 WSD 的优势；
- 打开 `gradient_checkpointing`，看吞吐下降多少、显存省了多少。

### 阶段 4：后训练（1~2 天）
读 `04-posttraining.md`。SFT → DPO → GRPO，观察每一步对输出风格的影响。

### 阶段 5：推理（重点，3~5 天）
读 `05-inference.md`，并跑：
```bash
uv run python benchmarks/inference_ablation.py --checkpoint out/tiny/last.pt --batch 8
```
**逐行对照表格理解每个优化的收益来源**。然后读源码：
`scheduler.py`（连续批处理）→ `engine.py`（组装）→ `kernels.py`（参考实现）
→ `triton_kernels.py`（高性能实现）→ `quantization/` → `speculative.py` → `cuda_graph.py`。

### 阶段 6：Agent（2 天）
读 `06-agentic.md`。先用 `EchoBackend` 把 Agent 逻辑跑通（不依赖模型），
再换成 `LocalEngineBackend` / `OpenAIBackend`。

### 阶段 7：对照 vLLM（1 天）
读 `07-vllm.md`。做两件事：
1. `scripts/export_hf.py` 把模型导出成 HF LLaMA 格式，
   体会"参数命名 / 权重布局 / 配置字段"这三类跨框架差异；
2. 在 Linux 上装 vLLM 跑 `benchmarks/compare_vllm.py`，
   把"教学实现 vs 生产实现"的差距量化出来（通常 5~20×）。
   然后按对照表去 vLLM 源码里找每一处差异的实现——这时你会发现
   生产系统的复杂度几乎全在"内核融合 + 并行 + 调度细节"上。

## 2. 贯穿全程的两个习惯

1. **每次改动都要有可度量的对比**：吞吐、显存、loss、过滤率——四者之一。
   本项目在 `benchmarks/` 与 `tests/` 里已经铺好了这些度量入口。
2. **先写 reference 实现，再优化**：`model/kernels.py` 就是这条原则的产物。
   没有朴素版本做对照，你永远不知道优化后的内核"快在哪、错没错"。

## 3. 延伸阅读（按主题）

- 数据：RefinedWeb (2023)、Dolma (2024)、FineWeb (2024)、DCLM (2024)
- 架构：LLaMA(2023)、Mistral 滑动窗口(2023)、DeepSeek-V2 MLA(2024)、
  DeepSeek-V3 MoE 无辅助损失(2024)、Qwen3(2025)、Gemma2 logit soft-capping(2024)、
  StreamingLLM Attention Sink(2023)、RetNet(2023)、Mamba(2023)
- 优化器：AdamW、Sophia(2023)、Lion(2023)、Muon / Moonlight(2024-2025)
- 推理：vLLM PagedAttention(2023)、FlashAttention(2022-2024)、
  Speculative Decoding(2023)、GPTQ(2022)、AWQ(2023)、SmoothQuant(2023)、QLoRA NF4(2023)
- 后训练：InstructGPT、DPO(2023)、DeepSeekMath GRPO(2024)、DAPO(2025)
- Agent：ReAct(2022)、Reflexion(2023)、Plan-and-Execute、Generative Agents 记忆(2023)、
  OpenAI Swarm / Handoff(2024)
