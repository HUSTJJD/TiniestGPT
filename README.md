# TiniestGPT

> **不是一个能用的大模型，而是一套"能跑通的大模型全链路教科书"。**
>
> 目标：用**极小参数量**（默认 ~25M）把大模型的**每一环**都亲手实现一遍——
> 数据 → 分词 → 架构 → 预训练 → 后训练 → 推理系统 → Agent 应用。
> 不追参数量，不追数据规模；追的是**覆盖面、先进性与工程正确性的可验证性**。

设计原则：

1. **全**：覆盖从原始文本到 Agent 应用的完整链路，一个环节都不外包给"调库"。
2. **精**：每个模块都用当前（2024–2026）业界前沿的做法，而不是教科书里的 2017 版 Transformer。
3. **可运行**：默认配置在**单张 RTX 3060 12GB**（甚至纯 CPU）上几分钟跑通端到端。
4. **可对照**：关键算法都提供 `reference`（朴素可读）与 `kernel`（高性能）两套实现，
   并带有数值一致性测试——先懂原理，再懂为什么快。
5. **可度量**：每个环节都有 benchmark / profile 入口，改动能立刻看到吞吐、显存、MFU 的变化。

---

## 一、全链路地图

| # | 环节 | 覆盖的关键技术 | 代码入口 |
|---|------|----------------|----------|
| 1 | **数据处理** | 归一化 / 规则过滤 / PII 脱敏 / MinHash+LSH 去重 / 质量打分（特征启发式 + 可训练判别器）/ BPE 训练 / 文档打包（无交叉污染）/ 内存映射分片 / **真实数据集下载（TinyStories / Shakespeare / WikiText-2）** | `tiniestgpt/data/` |
| 2 | **分词器** | 从零训练 BPE、GPT-2 风格预切分、special token、并行批量编码、encode/decode 往返测试 | `tiniestgpt/data/tokenizer/` |
| 3 | **模型架构** | Pre/Post RMSNorm(Sandwich)、RMSNorm/QK-Norm、RoPE+YaRN+mRoPE、GQA/MQA/MLA、SwiGLU、滑动窗口 + Attention Sink、稀疏 MoE（共享专家 + 无辅助损失负载均衡）、混合线性注意力层、z-loss | `tiniestgpt/model/` |
| 4 | **预训练** | AdamW / Muon / Sophia、WSD & 余弦调度、梯度裁剪 + NaN 守卫、bf16 混合精度、梯度检查点、DDP / FSDP、异步 checkpoint、MFU 统计 | `tiniestgpt/train/` |
| 5 | **后训练** | SFT（打包 + loss mask）、DPO/IPO、GRPO（组相对策略优化，含 KL 与优势归一化）、知识蒸馏（logit / hidden-state） | `tiniestgpt/posttrain/` |
| 6 | **推理系统（重点）** | PagedAttention、分块预填充、连续批处理调度、Prefix Caching、KV Cache 量化、Triton 内核、投机解码、CUDA Graph、OpenAI 兼容服务（**与 vLLM 逐模块对照**） | `tiniestgpt/inference/` |
| 7 | **量化** | INT8(W8A8 per-channel/per-token)、SmoothQuant、GPTQ(OBD/OBQ)、AWQ、NF4 + 双重量化、FP8(E4M3)、KV Cache INT8/FP8 | `tiniestgpt/inference/quantization/` |
| 8 | **Agentic 框架** | 类型化工具协议（自动生成 JSON Schema）、ReAct / Plan-and-Execute / Reflexion、分层记忆（工作/摘要/向量/情景）、上下文压缩、沙箱执行、多智能体编排（Supervisor / Blackboard / Handoff）、全链路 Trace | `tiniestgpt/agent/` |
| 9 | **CUDA 内核** | 手写 `.cu`：向量加法、Reduce 三连（原子加→共享内存→warp shuffle）、GEMM（朴素 vs 共享内存分块）、Softmax（朴素 vs online normalizer）、转置的 Bank Conflict 实验；JIT 编译 + 参考实现 + 微基准 | `tiniestgpt/kernels/` |
| 10 | **分布式并行** | 显存账本、张量并行（Column/Row 切分 + 模型手术）、序列并行、流水线并行（1F1B + 气泡分析）、自研 ZeRO-1/2/3；支持**单机模拟多卡** | `tiniestgpt/train/parallel/` + `common/memory_ledger.py` |
| 11 | **服务化与可观测** | 六项指标基准 + 回归门禁、Prometheus `/metrics`、结构化输出（约束解码）、Radix 前缀缓存、KV Swap/CPU offload、Prefill/Decode 解耦实验、Docker 部署 | `tiniestgpt/inference/{metrics,structured,radix_cache,swap}.py` |

---

## 二、目录结构

```
TiniestGPT/
├── docs/                       # 学习文档（每个环节一篇，含原理 + 论文索引 + 代码对照）
├── tiniestgpt/
│   ├── common/                 # 配置、日志、随机种子、Profiler、Registry、显存账本
│   ├── data/                   # 清洗 → 去重 → 打分 → 分词 → 打包 → 加载
│   ├── model/                  # 前沿小模型架构
│   ├── kernels/                # 手写 CUDA 内核（csrc/*.cu + JIT 加载 + 参考实现）
│   ├── train/
│   │   ├── ...                 # 预训练工程（优化器 / 调度 / 混合精度 / DDP+FSDP）
│   │   └── parallel/           # TP / SP / PP / ZeRO + 单机模拟多卡
│   ├── posttrain/              # SFT / DPO / GRPO / 蒸馏
│   ├── inference/              # 推理引擎 + AI Infra（重点）
│   │   ├── scheduler.py        # 连续批处理 / 分块预填充 / 前缀缓存 / 抢占
│   │   ├── radix_cache.py      # Radix 前缀树 + LRU 淘汰
│   │   ├── swap.py             # KV Cache 的 CPU 交换空间
│   │   ├── structured.py       # 结构化输出（约束解码）
│   │   ├── metrics.py          # Prometheus 指标
│   │   └── kernels/            # Triton / FlashAttention / INT4 融合 GEMM
│   └── agent/                  # Agentic 运行时
├── recipes/                    # 各规模/各阶段的 YAML 配方
├── scripts/                    # 便捷入口（setup / 环境自检 / CUDA 自检 / 剖析 / torchrun）
├── deploy/                     # Prometheus 抓取配置
├── Dockerfile / docker-compose.yml
├── tests/                      # 数值一致性 + 冒烟测试
└── benchmarks/                 # 吞吐/延迟/显存基准（含服务化基准与回归门禁）
```

---

## 三、快速开始（依赖全部由 uv 管理）

```bash
# Windows
.\scripts\setup.ps1

# Linux / macOS
bash scripts/setup.sh
```

脚本会：创建 uv 虚拟环境 → 从 **PyTorch 官方索引**装 CUDA 版 torch
→ 装 serving/dev 依赖并以 editable 方式安装本项目 → 跑 `scripts/verify_env.py` 自检。

想手动执行就三步（注意必须带 `--extra all`，否则不会装 fastapi/pytest）：

```bash
uv venv --python 3.14
uv sync --extra all          # 首次要下载约 2.6 GB 的 CUDA 版 torch，非交互终端下无进度条，请耐心等
uv run python -m tiniestgpt.cli info
```

> ⚠️ **Windows 上的已知坑**：uv 为 console script 生成 `.exe` trampoline 时，
> 若 `TEMP` 是 8.3 短名路径（如 `C:\Users\DAVIDS~1\AppData\Local\Temp`）会报
> `Failed to update Windows PE resources`。`scripts/setup.ps1` 已内置规避
> （把 TEMP 指向项目内 `.tmp`），手动执行时请先设置 `TEMP`/`TMP`。

# 2) 生成离线玩具语料 → 训练 BPE → 清洗/去重/打分 → 打包（无需联网）
python -m tiniestgpt.cli data --config recipes/data_toy.yaml

# 2b) 或直接用**真实数据集**（只需联网一次）
python -m tiniestgpt.cli datasets --probe          # 看内置数据集与体积
python -m tiniestgpt.cli data      --config recipes/data_tinystories.yaml   # TinyStories 19MB
python -m tiniestgpt.cli data      --config recipes/data_wikitext2.yaml     # WikiText-2（标准 benchmark）
python -m tiniestgpt.cli pretrain  --config recipes/pretrain_tinystories.yaml
python -m tiniestgpt.cli generate  --checkpoint out/tinystories/last.pt \
    --tokenizer data/tokenizer_ts.json --prompt "Once upon a time" --max-tokens 120

# 3) 预训练（单卡，约 3 分钟可见 loss 下降）
python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml

# 4) 推理引擎冒烟（连续批处理 + PagedAttention + 采样）
python -m tiniestgpt.cli generate --config recipes/pretrain_tiny.yaml \
    --checkpoint out/tiny/last.pt --prompt "Once upon a time"

# 5) 起 OpenAI 兼容服务
python -m tiniestgpt.cli serve --checkpoint out/tiny/last.pt --port 8000

# 6) 跑 Agent（本地引擎做大脑）
python -m tiniestgpt.cli agent --backend local --task "计算 2**10 + 3*(4+5) 并解释"

# 7) 基准：把推理每一项优化逐个打开，看吞吐变化
python benchmarks/inference_ablation.py --checkpoint out/tiny/last.pt

# 8) CUDA 内核自检（脱离 PyTorch 编译，最稳）与基准
python scripts/verify_cuda_kernels.py
python benchmarks/cuda_kernels.py

# 9) 服务化基准（六项指标 + 回归门禁）
python benchmarks/serving_bench.py --num-requests 32 --concurrency 8

# 10) 显存账本 / 分布式并行（无需多卡：单机模拟）
uv run python -c "from tiniestgpt.common.memory_ledger import *; \
print(format_report(estimate_training_memory(ModelShape(n_params=7_000_000_000, n_layers=32, \
dim=4096, hidden_dim=11008, n_heads=32, n_kv_heads=32, head_dim=128, seq_len=2048, batch_size=8), \
'adamw', world_size=8, zero_stage=3)))"
```

---

## 四、学习路线（建议顺序）

**阶段 0 · 地基**：`docs/00-roadmap.md` → 跑通 `data` 与 `pretrain` 两条流水线。
**阶段 1 · 数据**：理解"数据决定上限"，动手改清洗规则、观察去重率与质量分分布。
**阶段 2 · 架构**：逐个开关 `ModelConfig` 里的模块（GQA→MLA、Full→Window+Sink、Dense→MoE），
用 `tests/test_model.py` 保证数值不变性，理解每个结构"为什么被发明"。
**阶段 3 · 训练**：对比 AdamW / Muon、WSD / cosine，看 MFU 与显存，理解分布式通信重叠。
**阶段 4 · 后训练**：SFT → DPO → GRPO，观察对齐对 loss/生成质量的改变。
**阶段 5 · 推理（重头戏）**：按 `benchmarks/inference_ablation.py` 逐项叠加优化：
KV Cache → PagedAttention → 连续批处理 → Triton 内核 → 量化 → 投机解码 → CUDA Graph，
每步记录 tokens/s 与 p99 延迟。
**阶段 6 · Agent**：从零写工具、换 Planner、观察上下文压缩与多智能体编排。
**阶段 7 · CUDA 内核**（`docs/08-cuda.md`）：先跑 `scripts/verify_cuda_kernels.py`
看内核数值自检，再按 `01→05` 顺序读 `kernels/csrc/`，用 `benchmarks/cuda_kernels.py`
把"每一步优化省在哪"量化出来；最后用 Nsight 看 trace。
**阶段 8 · 分布式并行**（`docs/09-parallel.md`）：先用 `common/memory_ledger.py` 算账，
再用 `train/parallel/` 的**单机模拟模式**验证 TP/PP/ZeRO 的数学正确性，
最后 `scripts/train_torchrun.sh` 上真多卡。
**阶段 9 · 服务化**（`docs/10-serving.md`）：`benchmarks/serving_bench.py` 出六项指标 →
接 Prometheus `/metrics` → 开结构化输出 → 用决策树定位瓶颈。

---

## 五、学习文档

| 文档 | 内容 |
|---|---|
| [`docs/00-roadmap.md`](docs/00-roadmap.md) | 学习路线、每个阶段该做什么实验、延伸阅读 |
| [`docs/01-data.md`](docs/01-data.md) | 清洗 / 去重 / 质量 / BPE / 打包的原理与调参 |
| [`docs/02-architecture.md`](docs/02-architecture.md) | RMSNorm/RoPE/GQA/MLA/MoE/线性注意力为什么被发明 |
| [`docs/03-training.md`](docs/03-training.md) | 优化器、调度、混合精度、分布式、MFU |
| [`docs/04-posttraining.md`](docs/04-posttraining.md) | SFT / DPO / GRPO / 蒸馏 |
| [`docs/05-inference.md`](docs/05-inference.md) | **推理与 AI Infra（重点）**，含排错清单 |
| [`docs/06-agentic.md`](docs/06-agentic.md) | 工具协议、规划范式、记忆、多智能体 |
| [`docs/07-vllm.md`](docs/07-vllm.md) | **与 vLLM 的模块对照**：从教学实现到生产实现差在哪 |
| [`docs/08-cuda.md`](docs/08-cuda.md) | **CUDA 编程与算子优化**：GPU 架构、存储层次、5 个 kernel 实验、Nsight 工具链 |
| [`docs/09-parallel.md`](docs/09-parallel.md) | **分布式训练**：显存账本、ZeRO、TP / SP / PP、3D 并行怎么配 |
| [`docs/10-serving.md`](docs/10-serving.md) | **服务化与部署**：六项指标基准、Prometheus、结构化输出、PD 解耦、Docker |

## 六、验证状态

环境：`uv` + Python 3.14 + `torch 2.11.0+cu128`，RTX 3060 12GB / sm_86。

- `uv run python -m pytest tests`：**148 项测试，本机 130 通过 / 18 自动跳过**
  （分词往返、KV Cache 一致性、PagedAttention 与稠密等价、
  线性注意力并行/递推等价、投机解码分布一致性、量化误差、Agent 端到端、
  **TP/PP/ZeRO 与单卡等价**、**显存账本公式**、**Radix/swap/结构化输出**、**FlashAttention**）。
  跳过的 18 项需要 Triton（仅 Linux）或可用的 CUDA JIT 环境。
- **CUDA 内核自检**（`scripts/verify_cuda_kernels.py`，RTX 3060 / sm_86）：
  vector_add / reduce v0-v2 / gemm naive+tiled / softmax naive+online / transpose naive+padded
  **10/10 PASS**。
- 端到端冒烟已跑通：`data → pretrain → generate → agent → serve → quantize`。
- GPU 训练（bf16，20.2M 参数）：**~22,500 tok/s**，MFU 4%。
- **真实语料（TinyStories）实测**：19.4MB → 12,552 篇 → 4.88M tokens（BPE 3.99 B/token，
  padding 4.1%），全套流水线 **66 秒**；49M 参数模型训 300 步（约 7 分钟）
  loss 2.87→2.37、eval ppl **9.5**、11.5k tok/s、显存峰值 **8.3 GB**
  （RTX 3060 12GB 可跑，完整 3000 步约 70 分钟）。
- GPU 推理消融（`batch=8`，`benchmarks/inference_ablation.py`）：

| 优化项 | 吞吐 | 相对 L0 |
|---|---|---|
| L0 无 KV Cache | 15 tok/s | 1.00× |
| L1 稠密 KV Cache | 25 tok/s | 1.6× |
| L2 PagedAttention + 连续批处理 | 146 tok/s | **9.4×** |
| L3 + Prefix Caching（命中率 96%） | 140 tok/s | 9.0× |
| L4 + KV Cache INT8 | 120 tok/s | 7.8× |
| L5 + CUDA Graph | 142 tok/s | 9.2× |

（L4/L6 在本项目里是 PyTorch 参考实现，未做内核融合，因此收益被部分抵消——
这恰恰是"为什么生产系统必须自己写内核"的最好例证。）

## 七、与 vLLM 的关系

本项目**刻意重新实现**了 vLLM 的核心机制（而不是直接调库），
因为目标是让每个机制的原理可见。但为了避免"闭门造车"，另配了三层对照：

1. **`docs/07-vllm.md`** —— 逐模块对照表（本项目模块 ↔ vLLM 文件），
   以及一张"vLLM 做了而我们为了可读性省略了什么"的清单
   （张量并行、注意力后端矩阵、结构化输出、内核融合、Prometheus 指标 …）。
2. **导出成 HF LLaMA 格式** —— 我们的架构与 LLaMA 同构，
   导出后可直接被 transformers / vLLM / llama.cpp 加载：
   ```bash
   uv run python scripts/export_hf.py --checkpoint out/tiny/last.pt \
       --tokenizer data/tokenizer.json --out-dir out/tiny_hf
   ```
   > 有损特性（post-norm / qk-norm / 部分 RoPE / 混合层）会被显式警告，
   > 加 `--strict` 则直接报错——不静默导出一个数值不同的模型。
3. **对比基准** —— 同一份权重分别跑我们的引擎与 vLLM，把差距量化出来：
   ```bash
   uv run python benchmarks/compare_vllm.py --model-dir out/tiny_hf \
       --checkpoint out/tiny/last.pt --tokenizer data/tokenizer.json
   ```
   vLLM 未安装时会打印安装指引并优雅退出。
   > vLLM 只支持 Linux，且会约束 numpy 等基础依赖版本，
   > 因此**不**放进 extras；建议在独立环境里安装：
   > `uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm vllm`

## 八、硬件与依赖

| 项 | 说明 |
|---|---|
| 包管理 | **uv**（`uv.lock` 已提交，保证可复现） |
| Python | 3.14（`.python-version`） |
| torch | **cu128**（CUDA 12.8，由 `pyproject.toml` 的 `[tool.uv.sources]` 指定；PyPI 默认是 CPU 版） |
| 最低配置 | CPU + 8GB 内存（全链路可跑通，只是慢） |
| 推荐配置 | ≥8GB 显存的 NVIDIA GPU（本项目在 RTX 3060 12GB / sm_86 上验证） |
| **手写 CUDA 内核** | 额外需要 **CUDA Toolkit（nvcc）** + `ninja`（`uv sync --extra kernel`）+ Windows 上的 MSVC；缺一会自动回退到 PyTorch 参考实现，详见 `docs/08-cuda.md` 第 6 节 |

CUDA 版本切换：改 `pyproject.toml` 里 `[[tool.uv.index]]` 的 URL
（`cu128` → `cu130` 适配 Blackwell / RTX 50 系），然后 `uv sync --extra all`。

Triton 内核只支持 Linux：`uv sync --extra all --extra kernel`；
Windows 会自动回退到 PyTorch 参考实现（数值等价，仅速度差异）。
