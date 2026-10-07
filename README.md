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
| 1 | **数据处理** | 归一化 / 规则过滤 / PII 脱敏 / MinHash+LSH 去重 / 质量打分（特征启发式 + 可训练判别器）/ BPE 训练 / 文档打包（无交叉污染）/ 内存映射分片 | `tiniestgpt/data/` |
| 2 | **分词器** | 从零训练 BPE、GPT-2 风格预切分、special token、并行批量编码、encode/decode 往返测试 | `tiniestgpt/data/tokenizer/` |
| 3 | **模型架构** | Pre/Post RMSNorm(Sandwich)、RMSNorm/QK-Norm、RoPE+YaRN+mRoPE、GQA/MQA/MLA、SwiGLU、滑动窗口 + Attention Sink、稀疏 MoE（共享专家 + 无辅助损失负载均衡）、混合线性注意力层、z-loss | `tiniestgpt/model/` |
| 4 | **预训练** | AdamW / Muon / Sophia、WSD & 余弦调度、梯度裁剪 + NaN 守卫、bf16 混合精度、梯度检查点、DDP / FSDP、异步 checkpoint、MFU 统计 | `tiniestgpt/train/` |
| 5 | **后训练** | SFT（打包 + loss mask）、DPO/IPO、GRPO（组相对策略优化，含 KL 与优势归一化）、知识蒸馏（logit / hidden-state） | `tiniestgpt/posttrain/` |
| 6 | **推理系统（重点）** | PagedAttention、分块预填充、连续批处理调度、Prefix Caching、KV Cache 量化、Triton Flash 内核、投机解码、CUDA Graph、张量并行、OpenAI 兼容服务 | `tiniestgpt/inference/` |
| 7 | **量化** | INT8(W8A8 per-channel/per-token)、SmoothQuant、GPTQ(OBD/OBQ)、AWQ、NF4 + 双重量化、FP8(E4M3)、KV Cache INT8/FP8 | `tiniestgpt/inference/quantization/` |
| 8 | **Agentic 框架** | 类型化工具协议（自动生成 JSON Schema）、ReAct / Plan-and-Execute / Reflexion、分层记忆（工作/摘要/向量/情景）、上下文压缩、沙箱执行、多智能体编排（Supervisor / Blackboard / Handoff）、全链路 Trace | `tiniestgpt/agent/` |

---

## 二、目录结构

```
TiniestGPT/
├── docs/                       # 学习文档（每个环节一篇，含原理 + 论文索引 + 代码对照）
├── tiniestgpt/
│   ├── common/                 # 配置、日志、随机种子、Profiler、Registry
│   ├── data/                   # 清洗 → 去重 → 打分 → 分词 → 打包 → 加载
│   ├── model/                  # 前沿小模型架构
│   ├── train/                  # 预训练工程
│   ├── posttrain/              # SFT / DPO / GRPO / 蒸馏
│   ├── inference/              # 推理引擎 + AI Infra（重点）
│   └── agent/                  # Agentic 运行时
├── recipes/                    # 各规模/各阶段的 YAML 配方
├── scripts/                    # 便捷入口
├── tests/                      # 数值一致性 + 冒烟测试
└── benchmarks/                 # 吞吐/延迟/显存基准
```

---

## 三、快速开始

```bash
# 1) 安装
pip install -e ".[all]"

# 2) 生成离线玩具语料 → 训练 BPE → 清洗/去重/打分 → 打包（无需联网）
python -m tiniestgpt.cli data --config recipes/data_toy.yaml

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

---

## 五、硬件

- 最低：CPU + 8GB 内存（可跑通全链路，仅慢）。
- 推荐：单张 ≥8GB 显存的 NVIDIA GPU（本项目在 RTX 3060 12GB 上开发验证）。
- Triton 内核在 Linux 上可直接运行；Windows 下自动回退到 PyTorch 参考实现（数值等价，仅速度差异）。
