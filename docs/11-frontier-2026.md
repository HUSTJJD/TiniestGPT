# 大模型前沿技术雷达（2026 H2）

> 本文是 TiniestGPT 的"外部坐标系"。
> 目的不是追新，而是回答一个问题：**到 2026 年下半年，"全链路"这个词到底包含哪些技术？**
> 每张表最后一列给出本项目对该技术的判定（`已有` / `P0` / `P1` / `P2` / `不纳入`），
> 判定依据与落地方案见 [`docs/12-target-gap.md`](12-target-gap.md)。
>
> 资料基准：2026 年 7–9 月公开模型卡与技术报告
> （Qwen3.6、DeepSeek-V4、GLM-5.2、Kimi-K2.7、MiniMax-M2.7、Nemotron 3 Ultra、
> Gemma 4、Mistral Small 4、Llama 4、gpt-oss、Falcon-H1R、RWKV-7），
> 以及 vLLM / SGLang 2026 主线特性与 2026 年 post-training 综述。

---

## 0. 一句话看懂 2026 的架构主线

2026 年开放权重模型的演进可以压成五条线：

| # | 主线 | 具体表现 |
|---|---|---|
| 1 | **Attention 不再统一** | Full / Sliding / Sparse(DSA·CSA) / Compressed(HCA) / Linear(GDN·Mamba·RWKV) 六路并存，同一模型内按层混用 |
| 2 | **KV Cache 成为第一设计约束** | GQA → MLA → K=V 共享 → 跨层 KV Sharing → 递归固定状态 |
| 3 | **MoE 从"少算参数"走向硬件协同** | 细粒度专家(256~512) / 极小激活比(3~5%) / LatentMoE / FP4 专家 / Wide-EP |
| 4 | **训练目标直接服务推理** | MTP → 推测解码；原生低精度(NVFP4/MXFP4) → 硬件原生部署 |
| 5 | **多模态从外挂 Adapter 走向共同主干** | Early Fusion / 原生多模态 / Encoder-free |

**对教学项目的直接含义（2026-10 状态）**：本项目的 `layer_types` 混合架构已扩展到
`full / window / linear / gdn / mamba2`，补齐了 **Gated DeltaNet** 与 **Mamba-2/SSD**，
"训练目标服务推理" 这条线（**MTP**）也已落地。
**仍然缺失的是"稀疏/压缩检索注意力"整族**（DSA / CSA / HCA / IndexShare）——
线性注意力与稀疏注意力是两回事，见 §3.3，这是 v1.5 的第一优先项。

---

## 1. 数据层

| 技术 | 2026 状态 / 代表 | 本项目 | 判定 |
|---|---|---|---|
| 归一化 + 规则过滤 + 语言学启发式 | FineWeb / Dolma / RefinedWeb 标配 | `data/cleaning.py` | 已有 |
| MinHash + LSH 近似去重 | 工业标配 | `data/dedup.py` | 已有 |
| **Suffix Array 精确去重** | DCLM / RefinedWeb 主流（比 MinHash 更准） | 仅有 BloomFilter 精确去重 | P1 |
| 可训练质量判别器 | DCLM fastText / FineWeb classifier | `quality.py` 手写 numpy 逻辑回归 | 已有（教学足够） |
| **全局跨分片去重** | 大规模语料必须 | 仅单流水线内 | P2 |
| **Benchmark 去污染（decontamination）** | 2026 评测可信度的前提（n-gram 重叠剔除） | `data/decontaminate.py`（流水线 4b 阶段） | 已有 |
| **数据配比 / mixture 上采样** | Llama3 / Dolma 显式配比，退火阶段高质量上采样 | `data/mixture.py`（`MixtureSpec` + `AnnealingSchedule`） | 已有 |
| **合成数据飞轮** | 2026 后训练数据 55%+ 为合成 | 仅有 toy 语料生成器 | P1 |
| 课程学习 / 退火（annealing） | 训练末期换高质量数据 | 无 | P2 |
| 代码/数学数据专项处理 | 语法校验、去重、难度分级 | 无 | P2 |
| PII 脱敏 / 许可证追踪 | 合规标配 | 有 PII 正则 | 已有 |
| 数据消融实验框架 | "改一个环节看 loss 怎么变" | 有 `pipeline_report.json` | 已有 |
| 多模态数据（图文对齐/长视频 token 压缩） | 原生多模态前置 | 无 | 不纳入（见 §10） |

## 2. Tokenizer

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| 从零训练 BPE + GPT-2 预切分 + byte 回退 | 标配 | `data/tokenizer/` | 已有 |
| 并行批量编码 + encode/decode 往返测试 | 标配 | 已有 | 已有 |
| **词表规模 / 压缩率消融（tokenizer co-design）** | 影响 B/token 与训练成本 | 无 | P2 |
| 统一多模态离散 tokenizer | 2026 新方向 | 无 | 不纳入 |

## 3. 模型架构

### 3.1 主干与归一化

| 技术 | 2026 代表 | 本项目 | 判定 |
|---|---|---|---|
| Pre-RMSNorm / RMSNorm | 通用 | `norms.py` | 已有 |
| Sandwich（Pre+Post）Norm | Qwen3 / Gemma2 | `post_norm` | 已有 |
| QK-Norm | Qwen3 / Gemma4 / Qwen3.5 Gated Softmax Attn | `QKNorm` | 已有 |
| DynamicTanh (DyT) | 2025 新兴 | `norms.py` | 已有 |
| **mHC 流形约束超连接**（多残差流 + Sinkhorn-Knopp 双随机投影） | **DeepSeek-V4 (hc_mult=4)** | 无 | **P1（高教学价值）** |
| Logit soft-capping | Gemma2 | `attn_softcap` | 已有 |
| z-loss / depth-scaled init / tie embedding | 稳定化标配 | 已有 | 已有 |

### 3.2 位置编码

| 技术 | 2026 代表 | 本项目 | 判定 |
|---|---|---|---|
| RoPE | 通用 | `rope.py` | 已有 |
| YaRN 外推 | 通用 | 已有 | 已有 |
| mRoPE（多维拆分）/ partial RoPE | Qwen3-VL / Qwen3.6 / MiniMax | 已有 | 已有 |
| **NoPE + RoPE 交替（iRoPE）** | **Llama 4**（3 层 RoPE-local + 1 层 NoPE-global） | 无 | P2 |
| **Local/Global 层用不同 head_dim 与 rope_theta** | **Gemma 4**（local 256/1e4，global 512/1e6） | 无 | P2 |
| Attention Temperature Scaling（超长上下文推理） | Llama 4 Scout 10M | 无 | P2 |

### 3.3 注意力（**本项目最大缺口区**）

| 技术 | 2026 代表 | 本项目 | 判定 |
|---|---|---|---|
| MHA / GQA / MQA | 通用 | `attn_type` | 已有 |
| **MLA 低秩 KV 压缩** | DeepSeek / Kimi K2.7 / Mistral Small 4 | `mla.py` | 已有 |
| **MLA 权重吸收（weight-absorbed 推理形式）** | 生产必需（Kimi、DeepSeek 推理路径） | 未实现 | **P1** |
| 滑动窗口 Attention | gpt-oss(128)、Mistral、Gemma4(1024) | `attn_window` | 已有 |
| Attention Sink（可学习 sink bias） | gpt-oss / StreamingLLM | `attn_sinks` | 已有 |
| Local/Global 层交替（5:1） | Gemma 4 | `layer_types` 可表达 | 已有 |
| **K=V 共享（attention_k_eq_v）** | DeepSeek-V4 MQA、Gemma 4 global | 无 | P2 |
| **跨层 KV Sharing（producer/consumer）** | **Gemma 4 E2B**（35 层仅 15 个 KV producer，省 37.5%） | 无 | **P1** |
| **分组低秩输出投影（o_groups + LoRA rank）** | DeepSeek-V4（128 head / 16 组 / rank 1024） | 无 | P2 |
| **Gated DeltaNet（Delta Rule + 遗忘门）** | **Qwen3.6 / Qwen3-Next / Kimi Linear**（3:1 与 Full Attn 混合） | `model/sequence_mixer.py::GatedDeltaNet`（三形态） | 已有 |
| **Mamba-2 / SSD 选择性 SSM** | **Nemotron 3 Ultra、Falcon-H1R** | `model/sequence_mixer.py::Mamba2Mixer`（块内纯 GEMM） | 已有 |
| **RWKV-7 广义 Delta Rule（无 KV Cache）** | **RWKV-7 Goose** | 无 | P1 |
| **并行 Attention-SSM 混合块（同层通道切分）** | **Falcon-H1** | 无 | P1 |
| **可学习稀疏注意力（Lightning Indexer + Top-K，DSA）** | **DeepSeek-V3.2 / GLM-5.2（Top-2048）** | 无 | **P1** |
| **压缩稀疏 CSA / 重压缩 HCA** | **DeepSeek-V4**（m=4 选 1024 + m'=128 全读 + W=128 局部） | 无 | P1 |
| **IndexShare（跨层复用 Top-K 索引）** | **GLM-5.2**（每 4 层一次 Indexer，1M 上下文省 2.9× FLOPs） | 无 | P2 |
| Chunk 并行 Scan / WY 表示（线性注意力训练形态） | GDN / Mamba 训练必需 | 无（Retention 有并行形态） | P1 |

> 2026 关键认知：**线性注意力 ≠ 稀疏注意力**。
> 线性 = 历史压进固定状态 S，decode O(1)，但精确回看弱；
> 稀疏 = 仍存历史，每个 query 选 k 个位置，decode O(k)，保留内容检索但付 Indexer + Top-K 代价。
> 本项目目前只有"全量 + 窗口 + Retention 线性"三档，缺"稀疏检索"这一整族。

### 3.4 FFN 与 MoE

| 技术 | 2026 代表 | 本项目 | 判定 |
|---|---|---|---|
| SwiGLU / GEGLU / ReGLU / ReLU² | 通用 | `mlp.py` | 已有 |
| gate+up 融合单 GEMM | 通用 | 已有 | 已有 |
| 稀疏 MoE（Top-K + 共享专家） | 通用 | `moe.py` | 已有 |
| Sigmoid 路由 + 负载校正 bias（无辅助损失） | **MiniMax-M2.7 / DeepSeek-V3** | `moe_score_func` + `moe_bias_lr` | 已有 |
| Switch aux loss | 经典 | `moe_aux_coef` | 已有 |
| 容量因子 / token drop / 路由监控 | 通用 | 已有 | 已有 |
| **细粒度专家 + 极小激活比** | Kimi(384 experts)、MiniMax(256, Top-8, 4.3%)、Nemotron(512, Top-22) | 支持配置，无配方 | P2 |
| **LatentMoE（先降维再路由，All-to-All payload 降 4×）** | **Nemotron 3 Ultra** | 无 | P1 |
| **Hash MoE（前几层静态哈希路由做 warmup）** | **DeepSeek-V4 前 3 层** | 无 | P2 |
| **Per-Layer Embedding (PLE)** | **Gemma 4 E2B/E4B**（用权重容量换 Dense GEMM FLOPs） | 无 | P2 |
| **Clamped SwiGLU（激活 clamp 保低精度稳定）** | **gpt-oss (swiglu_limit=7.0)** | 无 | P2 |
| **MoE 分组 GEMM / 融合 permute-unpermute** | 生产推理必需 | 无（Python 循环实现） | **P1** |

### 3.5 训练目标 / 效率结构

| 技术 | 2026 代表 | 本项目 | 判定 |
|---|---|---|---|
| **MTP 多 token 预测（训练目标 + 推理 draft head）** | **DeepSeek / Kimi / MiniMax(3 步) / GLM-5.2 / Nemotron(MTP Boosting)** | `model/mtp.py` + `inference/mtp_spec.py`（自草稿） | 已有 |
| 知识蒸馏（logit / hidden-state） | 通用 | `distill.py` | 已有 |
| 早退 / 层跳过 | 研究阶段 | 无 | 不纳入 |

## 4. 预训练工程

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| AdamW / Muon / Lion / Sophia | Muon 已成旗舰标配（DeepSeek-V4、Kimi K2） | `optim.py` 四种齐全 | 已有 |
| **MuonClip / QK-Clip（约束 attention logit 防爆）** | **Kimi K2（15.5T token 零 loss spike）** | `train/qk_clip.py`（逐 head 缩放，兼容 GQA） | 已有 |
| **分布式 Muon（optimizer state sharding + 正交化通信）** | 2026 DMuon 等 | 单机 Muon | P1 |
| WSD / cosine / 可随时起 decay | WSD 主流 | `lr_sched.py` | 已有 |
| bf16 混合精度 + fp32 主权重 | 通用 | `engine.py` | 已有 |
| **FP8 训练（block-wise scaling / 主权重 / 随机舍入）** | **DeepSeek-V3/V4** | 仅推理侧 FP8 量化 | **P1** |
| **原生 FP4 / NVFP4 / MXFP4 预训练** | **Nemotron 3 Ultra、gpt-oss(MXFP4 MoE)、DeepSeek-V4(FP4 专家)** | 无 | P2 |
| 梯度裁剪 + NaN 守卫 | 通用 | 已有 | 已有 |
| 梯度检查点 | 通用 | 已有 | 已有 |
| DDP / FSDP | 通用 | `distributed.py` | 已有 |
| TP / SP / PP(1F1B) / ZeRO 0-3 | 通用 | `train/parallel/`（单进程模拟） | 已有 |
| **Context Parallel / Ring Attention** | 1M 上下文训练必需 | 无 | P2 |
| **融合交叉熵 / chunked CE（省 logits 显存）** | Liger / 2026 主流显存优化 | 无 | P1 |
| **EMA / 权重平均** | 常见 | 无 | P2 |
| **异步 + 分布式 checkpoint、故障自愈重启** | 大规模训练必需 | 有 checkpoint，无异步/自愈 | P2 |
| MFU 统计 / profiler / 显存账本 | 通用 | `common/profiler.py` + `memory_ledger.py` | 已有 |
| **loss spike 自动检测与回滚** | 2026 训练平台标配 | 无 | P2 |

## 5. 后训练（**本项目第二大缺口区**）

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| SFT（packing + prompt loss mask） | 标配 | `sft.py` | 已有 |
| DPO / IPO / hinge | 标配 | `dpo.py` | 已有 |
| GRPO（组相对优势 + DAPO clip-higher + KL k1/k3） | **2026 事实标准**（DeepSeek-R1、Qwen3、Kimi K2） | `grpo.py` | 已有 |
| **奖励模型 RM（pairwise BT loss）** | 混合奖励（70% RM + 30% 规则）是最佳实践 | `posttrain/reward.py::RewardModel` + `HybridReward` | 已有 |
| **RLVR 可验证奖励（规则/单测/答案匹配）** | **o1 / R1 范式，2026 主线** | `posttrain/reward.py::RuleReward`（5 种规则） | 已有 |
| **PRM 过程奖励模型** | Let's Verify Step by Step / PRM800K | 无 | P2 |
| **Rollout 引擎（采样循环 + 训练引擎权重同步）** | verl / OpenRLHF / trl 核心 | `posttrain/rollout.py` + `grpo_trainer.py`（闭环） | 已有 |
| PPO + critic + GAE | 经典，教学价值高 | 无 | P1 |
| RLOO / REINFORCE++ | 2025-2026 常见变体 | 无 | P2 |
| KTO / ORPO / SimPO | DPO 修补族（Kimi K2 用 SimPO+DPO） | 无 | P1 |
| **在线/迭代 DPO（重新采样偏好对防分布漂移）** | 2026 取代离线 DPO | 无 | P1 |
| **拒绝采样 / best-of-n / 迭代 SFT-RL** | Llama 3.1 405B 五轮迭代 | 无 | P1 |
| **Agentic RL（多轮工具交互轨迹 + 规则奖励）** | **2026 SWE-bench / τ-bench SOTA 范式** | 无 | **P1** |
| RLAIF / 宪法反馈 | Anthropic 主线 | 无 | P2 |
| 奖励塑形（长度/格式/重复惩罚）+ reward hacking 检测 | 工程必需 | 无 | P1 |
| 知识蒸馏（logit / hidden） | 通用 | 已有 | 已有 |
| Thinking mode / 可控推理预算 | Qwen3 / Mistral Small 4 | 无 | P2 |

> 2026 共识：**SFT + 偏好对齐 + GRPO/RLVR + 安全对齐** 是四件套。
> 本项目目前有 SFT、DPO、GRPO 的**损失函数**，但没有"采样 → 打分 → 更新"的**闭环**，
> 也就是说 GRPO 目前是"半个"——这是最需要优先补的洞。

## 6. 推理引擎

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| PagedAttention + 连续批处理 | 通用 | `scheduler.py` + `kv_cache.py` | 已有 |
| Chunked Prefill | 通用 | 已有 | 已有 |
| 前缀缓存（块哈希 + Radix 树 + LRU） | vLLM / SGLang RadixAttention | `PrefixCache` + `radix_cache.py` | 已有 |
| 抢占（重算 / CPU swap） | 通用 | `swap.py` | 已有 |
| 采样全家桶（temp/top-p/top-k/min-p/typical-p/惩罚） | 通用 | `sampler.py` | 已有 |
| 结构化输出（约束解码） | 2026 必备（工具调用前提） | `structured.py` 自研 JSON 前缀状态机 | 已有 |
| **GBNF / DFA 预编译任意 schema 约束解码** | vLLM/SGLang 用 xgrammar/llguidance | 仅 JSON 子集 | P1 |
| 投机解码（draft-target 接受-拒绝 + bonus token） | 通用 | `speculative.py` | 已有 |
| **MTP / EAGLE-3 / Medusa 自草稿（无需独立 draft 模型）** | **2026 主力**（PayPal 实测 EAGLE-3 吞吐 +20~50%，一卡抵两卡） | `inference/mtp_spec.py`（L7 档报 τ） | 已有 |
| **树状投机 / 多路验证** | EAGLE-2/3、Medusa | 无 | P2 |
| **PD 解耦（prefill/decode 分离 worker + KV 传输）** | 2025-2026 成熟范式，vLLM/SGLang 已支持 | 只有 `benchmarks/pd_disaggregation.py` 对比实验 | **P1** |
| CUDA Graph（分桶 + 静态缓冲） | 通用 | `cuda_graph.py` | 已有 |
| 量化：RTN / GPTQ / AWQ / SmoothQuant / NF4+双重量化 / FP8 / KV 量化 | 通用 | `quantization/` 齐全 | 已有 |
| **NVFP4 / MXFP4 微缩放（block scale）量化** | **gpt-oss、Nemotron、Mistral Small 4** | 无 | P2 |
| **多状态 Cache Manager**（Paged KV / Ring KV / 递归 State / 压缩池 / 共享 KV slot） | **2026 混合架构的必需品** | 只有 Paged + Dense | **P1** |
| **Layer-aware 算子分派**（按层类型路由到不同 kernel/状态） | 2026 混合架构必需品 | `dispatch.py` 只按 backend 分派 | **P1** |
| **Prefix Cache 对递归状态的快照支持** | GDN/Mamba 前缀复用必需 | 无 | P2 |
| Triton Paged Decode / RMSNorm / SwiGLU / FlashAttention / INT4 GEMM | 通用 | `inference/kernels/` | 已有 |
| **Flash-Decoding / split-KV、Paged Prefill kernel** | 长上下文必需 | 无 | P1 |
| **CUDA 版 PagedAttention** | vLLM 核心 | 仅 Triton + PyTorch ref | P2 |
| **MoE 推理：Grouped GEMM + Expert Parallel + All-to-All** | 通用 | 无 | P1 |
| **LoRA 多租户热插拔** | 2026 服务标配 | 无 | P2 |
| **Sleep mode / 权重卸载（训推切换）** | RL 后训练必需（vLLM sleep level 1/2） | 无 | P1 |
| **分层 KV 缓存（GPU→CPU→SSD，HiCache / LMCache）** | 2026 长上下文降本主线 | 只有 CPU swap | P2 |
| Beam search / n-best | 通用 | 无 | P2 |
| 多模态前缀缓存（图像哈希入 key） | vLLM 2026 | 无 | 不纳入 |

## 7. 服务化与部署

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| OpenAI 兼容 HTTP + SSE 流式 | 通用 | `server.py` | 已有 |
| Prometheus 指标 | 通用 | `metrics.py` | 已有 |
| 六项服务化指标基准 + 回归门禁 | 通用 | `benchmarks/serving_bench.py` | 已有 |
| 结构化输出 | 通用 | 已有 | 已有 |
| Docker / compose / Prometheus 抓取配置 | 通用 | `deploy/` | 已有 |
| **语义路由 / 多模型路由（vLLM Semantic Router）** | 2026 新层 | 无 | P2 |
| **SLO-aware 调度（优先级 / 公平性 / TTFT-TPOT 权衡）** | 生产必需 | 无 | P1 |
| **三层缓存（网关响应缓存 + 提示词缓存 + 引擎前缀缓存）** | 2026 降本主线 | 无引擎外缓存 | P2 |
| 自动扩缩容 / 成本追踪 / 每 token 成本 | 生产必需 | 无 | P2 |
| gRPC / 多机 TP 推理 | 生产必需 | 无 | P2 |

## 8. CUDA / 内核

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| 教学五连（vector add / reduce 三连 / GEMM 分块 / online softmax / bank conflict） | 入门标配 | `kernels/csrc/` 10/10 PASS | 已有 |
| Triton 内核（paged decode / rmsnorm / swiglu / flash / int4 gemm） | 通用 | `inference/kernels/` | 已有 |
| JIT 编译 + 自动降级 + 参考实现 + 微基准 | 教学设计 | `kernels/loader.py` | 已有 |
| **Chunk 并行 Scan / WY 表示内核（GDN / Mamba）** | 2026 混合架构核心 kernel | 无 | P1 |
| **MoE 分组 GEMM 内核** | 核心 | 无 | P1 |
| **融合算子：RoPE、残差+Norm、SwiGLU 融合、fused permute** | 生产必需 | 部分（Triton SwiGLU/RMSNorm） | P1 |
| **Attention 反向内核** | 训练必需 | 无（用 PyTorch autograd） | P2 |
| **FP8 GEMM / 微缩放内核** | 2026 主力 | 无 | P2 |
| **Hopper / Blackwell 特性（WGMMA、tcgen05、TMA、异步流水线）** | 2026 算子开发前沿 | 无 | P2 |
| Nsight trace 实验 | 文档已覆盖 | `docs/08-cuda.md` | 已有 |

## 9. Agent 与应用

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| ReAct / Plan-and-Execute / Reflexion | 经典三范式 | `planner.py` | 已有 |
| 类型化工具协议（注解 → JSON Schema） | 通用 | `tools.py` | 已有 |
| 分层记忆（工作 / 摘要 / 向量 / 情景） | 通用 | `memory.py` | 已有 |
| 上下文压缩 / token 计数 | 通用 | `context.py` | 已有 |
| 多智能体（Supervisor / Blackboard / Handoff） | 通用 | `multi_agent.py` | 已有 |
| 全链路 Trace（JSONL 可重放） | 通用 | `observability.py` | 已有 |
| **MCP（Model Context Protocol）客户端/服务端** | **2026 工具层事实标准** | `agent/mcp.py`（stdio + http，含内置 echo server） | 已有 |
| **A2A（Agent-to-Agent）协议** | 2026 多智能体互操作 | 无 | P2 |
| **真沙箱（容器/进程隔离、资源限额、网络白名单）** | 2026 长程 Agent 前提 | `agent/sandbox.py`（子进程 + CPU/内存限额 + 禁网 + 持久化目录） | 已有 |
| **Context Engineering（文件系统 + Shell + 持久化任务状态 + 技能库）** | **2026 长程 Agent 主线** | 无（只有内存态记忆） | **P1** |
| **长任务可恢复 / 断点续跑 / checkpoint** | 长程 Agent 必需 | 无 | P1 |
| **子智能体上下文隔离 + 并行（Agent Swarm）** | Anthropic/Kimi 编排范式 | 有编排，无上下文隔离 | P1 |
| **成本护栏（token 预算、循环/挂起检测、超时）** | 生产必需 | 无 | **P1** |
| **HITL 人工审批 + 权限分级** | 企业落地必需 | 无 | P2 |
| **技能库 / 自我改进（失败轨迹 → 沉淀技能）** | 2026 长程 Agent 主线 | 无 | P2 |
| 真向量库 / 图记忆 / 记忆巩固 | 常见 | 哈希嵌入 | P2 |
| Computer Use / 浏览器操作 | 2026 通用 Agent 能力 | 无 | P2 |

## 10. 多模态

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| Early Fusion（视觉 token 与文本共同进主干） | Llama 4 / Qwen3.6 / Kimi K2.x | 无 | 不纳入（见 §11 边界） |
| Conv3D patch embedding + spatial merge + mRoPE 时空 | Qwen3.6 视频 | 无 | 不纳入 |
| Encoder-free（原始 patch / 音频块直接投影） | Gemma 4 12B | 无 | 不纳入 |
| 视频 token 压缩 / 长视频采样 | 2026 难点 | 无 | 不纳入 |

## 11. 评测与可观测（**本项目第三大缺口区**）

| 技术 | 2026 状态 | 本项目 | 判定 |
|---|---|---|---|
| 训练 loss / perplexity | 基础 | 有 | 已有 |
| 吞吐 / 延迟 / 显存 / MFU 基准 | 基础 | `benchmarks/` | 已有 |
| **标准能力评测接入（HellaSwag / ARC / MMLU 子集 / Lambada / GSM8K 子集）** | 教学项目验证模型"真学会了"的唯一手段 | `eval/`（算术 / 长程召回 / JSON / 完形，含 loglikelihood 与规则两种打分） | 已有 |
| **评测去污染** | 见 §1 | `data/decontaminate.py` | 已有 |
| **生成质量评测（重复率、多样性、KL 参考分布）** | 基础 | 无 | P1 |
| **Agent 任务评测（成功率、Pass^k 多轮一致性、成本）** | 2026 从"能不能用"到"靠不靠谱" | 无 | **P1** |
| **质量回归门禁（不只性能门禁）** | 改动可见 | 只有性能门禁 | P1 |
| 量化/低精度精度损失评测 | 必需 | 有量化误差测试 | 已有 |

---

## 12. 边界：明确不纳入 target

为了保持"单卡 3060 · 几分钟跑通 · 小而全"的定位，以下技术**主动放弃**：

| 不纳入 | 原因 |
|---|---|
| 多模态（视觉/音频编码器、视频） | 需要预训练视觉数据与算力，与"文本全链路"定位冲突；可作为独立分支项目 |
| 真多机 / 真多卡通信（保留单进程模拟） | 硬件约束；教学目标是"数学正确性可验证"，模拟已达成 |
| 追求参数量与 SOTA 效果 | 项目目标是覆盖度与可验证性，不是跑分 |
| 原生 FP4 预训练、Hopper/Blackwell 专用指令 | 硬件约束（sm_86），仅保留 CPU/GPU 通用路径 |
| 商用级多租户、自动扩缩容、语义路由 | 属于平台工程，与"机制可见"冲突 |
| 生产级 RL 集群（Ray / verl 规模） | 保留单机可跑的最小闭环 |

这套边界的意义：**target 不是"追平 vLLM + Megatron"，而是"让 2026 的每一个关键机制，都能在 25M 参数上被看见、被度量、被验证"。**
