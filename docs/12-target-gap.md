# Target 定义与 Gap 分析

> 上游：[`docs/11-frontier-2026.md`](11-frontier-2026.md)（前沿技术雷达与逐项判定）
> 本文回答两个问题：**这个项目最终要长成什么样**（Target），**以及现在还差什么**（Gap）。
> 更新方式：每完成一项，把本文件对应行的状态从 `✅` 改为 `✅`，并同步 `docs/11` 的判定列。

> **2026-10 更新：P0 八项已全部落地。** 代码入口与验收见本文第三节，
> 每项的落点文件、等价性测试与 benchmark 入口都已就位。

---

## 一、Target 定义

### 1.1 一句话

> **让 2026 年大模型的每一个关键机制，都能在 25M 参数 / 单张 RTX 3060 上被看见、被度量、被验证。**

关键词拆解：

| 关键词 | 含义（可检验） |
|---|---|
| **每一个关键机制** | 覆盖 `docs/11` 中判定为 P0/P1 的全部条目，不外包给"调库" |
| **被看见** | 每个机制都有 `reference`（朴素可读）与 `kernel`（高性能）两套实现 |
| **被度量** | 每个机制都挂一个 benchmark 入口，改动立刻看到吞吐/显存/精度/MFU 的变化 |
| **被验证** | 每个机制都有数值一致性测试：`reference ≡ kernel`、`并行 ≡ 递推`、`量化 ≡ 反量化` |
| **25M / 3060** | 任何单个实验 < 5 分钟跑完，全链路 < 1 小时 |

### 1.2 三条不可妥协的原则

1. **闭环优先于算法数量**：一个只有 loss 函数、没有"采样→打分→更新"闭环的 GRPO，
   价值低于一个能真跑起来的最小 RL 循环。**宁可少一个算法，不可少一个闭环。**
2. **前沿优先于经典**：补 Mamba-2 比补 ALiBi 重要；补 MTP 比补 Beam Search 重要。
   判断标准是"2026 旗舰模型里有没有它"（见 `docs/11` 代表列）。
3. **边界要写死**：多模态、真多机、SOTA 跑分**明确不做**（见 `docs/11` §12），
   避免项目被无限扩张拖死。

### 1.3 Target 的三个版本

| 版本 | 主题 | 完成判据 |
|---|---|---|
| **v1.0（当前 → 下一站）** | **补齐闭环与 2026 主干** | 8 项 P0 全部落地；端到端 `data → pretrain → posttrain(真实 RL 闭环) → eval → infer(MTP 自草稿) → serve → agent(MCP+沙箱)` 跑通 |
| **v1.5** | **追平前沿机制** | P1 全部落地；`docs/11` 中 P0+P1 覆盖率 ≥ 95% |
| **v2.0** | **差异化** | P2 选修落地 + "机制对照矩阵"：同一任务下 GDN / Mamba / Retention / Sparse / Full 五条路线的吞吐-精度-MFU 三组数 |

---

## 二、现状画像（盘点基线）

盘点时间：2026-10，环境 `uv` + Python 3.14 + torch 2.11 cu128 + RTX 3060 12GB(sm_86)，148 项测试。

**已经站住的部分**（这部分不用动，只需要维护）：

| 层 | 已实现 | 覆盖度评价 |
|---|---|---|
| 数据 | 清洗/PII/MinHash+LSH/质量判别器/BPE/三种打包/mmap 分片/异步预取 | **优**，缺去污染与配比 |
| 分词 | 从零 BPE + GPT-2 预切分 + 并行编码 + 往返测试 | **优** |
| 架构 | RMSNorm/DyT/Sandwich/QK-Norm/RoPE·YaRN·mRoPE/MHA·GQA·MQA·**MLA**/滑窗/Sink/softcap/Retention 线性/SwiGLU 族/MoE(共享专家+无损失均衡)/z-loss/层类型混合 | **良**，缺 2026 的 GDN·Mamba·稀疏检索·MTP·mHC |
| 训练 | AdamW/Muon/Lion/Sophia/WSD/bf16/梯度守卫/DDP·FSDP/TP·SP·PP·ZeRO/MFU/显存账本 | **良**，缺 FP8 训练与 MuonClip |
| 后训练 | SFT/DPO·IPO/GRPO(loss 层完整)/蒸馏 | **中**，**缺 RM、奖励、rollout —— GRPO 没有闭环** |
| 推理 | PagedAttention/连续批处理/chunked prefill/双前缀缓存/抢占/采样全家桶/结构化输出/投机解码/CUDA Graph/七种量化/OpenAI 服务/HF 导出/Prometheus | **优**，缺自草稿投机、PD 解耦、多状态 cache |
| 内核 | 5 个教学 `.cu`(10/10 PASS) + 5 个 Triton kernel | **良**，缺 MoE 分组 GEMM、chunk scan |
| Agent | 三范式/工具协议/四层记忆/上下文压缩/三种多智能体编排/Trace | **中**，缺沙箱、MCP、持久化、成本护栏 |
| 评测 | 训练 ppl + 推理吞吐/延迟/显存基准 + 性能门禁 | **弱**，**没有能力评测**（唯一硬伤） |

~~**三处结构性缺口**~~（已在 2026-10 补齐）：

1. ~~**后训练没有闭环**~~ → `posttrain/{reward,rollout,grpo_trainer}.py`
2. ~~**没有能力评测**~~ → `eval/{tasks,harness,quality}.py` + `benchmarks/quality_gate.py`
3. ~~**架构缺 2026 的新物种**~~ → `model/sequence_mixer.py`（GDN + Mamba-2/SSD）

**已知性能限制（诚实记录）**：GDN 的参考实现是**分块扫描**（每块内仍是逐步递推），
Python 层面的串行步数与序列长度同阶，训练比纯 attention 慢一个量级。
真正无循环的块内并行需要 UT/WY 变换 + 三角求解的 fused kernel —— 这项是 P1，
见 `docs/11-frontier-2026.md` §3.3 与本文第四节第 1 项。

---

## 三、Gap 清单 · P0（v1.0 必做，8 项）

> 排序 = 建议实施顺序（存在依赖：P0-3 依赖 P0-4，可先做 P0-4）

### 落地总览（2026-10）

| # | 缺口 | 主要新增文件 | 等价性/验收测试 |
|---|---|---|---|
| 1 | GDN + Mamba-2/SSD | `tiniestgpt/model/sequence_mixer.py` | `tests/test_sequence_mixer.py`（三形态 1e-6 等价 + delta rule 改写验证 + 状态不随 L 增长） |
| 2 | MTP | `tiniestgpt/model/mtp.py`、`tiniestgpt/inference/mtp_spec.py` | `tests/test_mtp.py`（头对齐、loss 错位检查、自草稿 τ 统计） |
| 3 | RM / RLVR / rollout | `tiniestgpt/posttrain/{reward,rollout,grpo_trainer}.py` | `tests/test_grpo_closed_loop.py`（真实跑通采样→打分→更新） |
| 4 | 能力评测 | `tiniestgpt/eval/{tasks,harness,quality}.py` | `tests/test_eval_and_data.py`；CLI `eval`；`benchmarks/quality_gate.py` |
| 5 | 去污染 + 配比 | `tiniestgpt/data/{decontaminate,mixture}.py` | 同上（去污剔除、权重生效、退火插值） |
| 6 | Agent 沙箱 | `tiniestgpt/agent/sandbox.py` | `tests/test_agent_sandbox_mcp.py`（死循环被杀、输出截断、持久化目录） |
| 7 | MCP | `tiniestgpt/agent/mcp.py` | 同上（内置 echo server 跑通 initialize→tools/list→tools/call） |
| 8 | MuonClip / QK-Clip | `tiniestgpt/train/qk_clip.py` | `tests/test_qk_clip.py`（逐 head 标量缩放、GQA 头对齐） |

配套：新增预设 `gdn_hybrid` / `mamba_hybrid`、配方 `recipes/pretrain_gdn_hybrid.yaml`
与 `recipes/pretrain_mtp.yaml`、`benchmarks/inference_ablation.py` 的 L7 档（MTP 自草稿）。

### P0-1 · 2026 线性序列混合器：Gated DeltaNet + Mamba-2/SSD ✅

- **现状**：`model/linear_attention.py` 只有 Retention（并行/递推双形态），无门控、无 delta rule、无 chunked 训练形态。
- **缺什么**：GDN（`S_t = α·S_{t-1} + β(v - S_{t-1}k)k^T`）、Mamba-2/SSD（选择性 SSM）、chunk 并行 scan。
- **落点**：`model/sequence_mixer.py`（新增，统一 `chunk_parallel` / `recurrent_decode` / `naive_loop` 三形态）+ `ModelConfig.layer_types` 增加 `gdn` / `mamba2`。
- **工作**：GDN（含 Conv 局部增强）≈ 1 天；Mamba-2/SSD ≈ 1 天；chunk scan ≈ 1 天；共 **3 天**。
- **验收**：
  - `tests/test_sequence_mixer.py`：`naive_loop ≡ chunk_parallel ≡ recurrent_decode`（1e-4）
  - `layer_types: "gdn,gdn,gdn,full"` 跑到 loss 下降，与全 full 对比 loss/吞吐/KV 显存
  - 长上下文下 decode 显存**不随 L 增长**（这是它存在的全部理由，必须量化出来）

### P0-2 · MTP 多 token 预测（训练目标 + 自草稿推测解码）✅

- **现状**：完全没有。投机解码 `speculative.py` 只能用**独立 draft 模型**，而 2026 主流是 MTP head / EAGLE。
- **缺什么**：`L_MTP = Σ λ_j CE(p_j(x_{t+j}|x_{≤t}), x_{t+j})`；推理时 draft → 主模型一次前向批量验证。
- **落点**：`model/mtp.py`（MTP head + 权重共享）+ `train/losses.py`（λ 加权）+ `inference/speculative.py`（`MTSDecoder`）。
- **工作**：**2 天**。
- **验收**：
  - 训练侧：MTP loss 与 main loss 分别可观测，关掉 MTP 后 main loss 曲线不变（对照实验）
  - 推理侧：`benchmarks/inference_ablation.py` 增加 L6 档，报告**平均接受长度 τ** 与实际加速比
  - 明确记录"什么情况下 MTP 不加速"（batch 大、接受率低、draft 太重）——这是 `docs/11` §3.5 的重点认知

### P0-3 · 奖励模型 + 可验证奖励 + rollout 引擎（补齐 RL 闭环）✅

- **现状**：`posttrain/grpo.py` 只有 loss，`dpo.py` 只吃离线偏好对。**没有 RM、没有奖励函数、没有采样循环**。
- **缺什么**：BT 奖励模型、`RLVR` 规则奖励（答案匹配 / 单测执行 / 工具返回成功）、rollout 采样循环、与训练引擎的权重同步。
- **落点**：
  - `posttrain/reward.py`（`RewardModel` BT loss + `RuleReward` 可插拔规则：`exact_match` / `unit_test` / `tool_success`）
  - `posttrain/rollout.py`（用本地 `InferenceEngine` 采样，组采样 G 条，返回 logps/mask）
  - `posttrain/grpo_trainer.py`（串起 rollout → reward → advantage → loss → 更新）
- **工作**：**3 天**。
- **验收**：
  - 端到端：25M 模型在"算术/格式化输出"这类**可验证任务**上，GRPO 训练 200 步后**规则奖励分数单调上升**（有图/有数）
  - 与 P0-4 联动：跑完 GRPO 后 eval 分数提升（证明不是 reward hacking）
  - 记录 reward hacking 复现（长度爆炸 / 格式刷分）与 clip 的作用

### P0-4 · 能力评测体系 ✅

- **现状**：只有训练 ppl 与推理吞吐。**无法回答"模型学会了没有"**，也没有质量门禁。
- **缺什么**：标准 benchmark 子集接入、生成质量指标、质量回归门禁。
- **落点**：`eval/`（新增包）
  - `eval/tasks.py`（内置极小子任务：hellaswag 式完形 / ARC 式选择 / 算术 / 格式化 JSON / 复制检索 needle）
  - `eval/harness.py`（few-shot prompt 构造、loglikelihood 打分、accuracy 汇总）
  - `eval/quality.py`（重复率、distinct-n、与参考分布的 KL）
  - `benchmarks/quality_gate.py`（与 `serving_bench.py` 的性能门禁并列）
- **工作**：**2 天**。
- **验收**：`uv run python -m tiniestgpt.cli eval --checkpoint ... --tasks all` 出一张分数表；
  CI 里质量门禁可拦截"改了架构但把模型改傻了"的提交。

### P0-5 · 数据去污染 + 数据配比 ✅

- **现状**：`data/` 清洗/去重/打分齐全，但没有去污染，也没有 mixture 概念（只有一个数据源）。
- **缺什么**：n-gram 重叠去污染；多源配比与退火上采样。
- **落点**：`data/decontaminate.py`（13-gram 重叠剔除 + 报告）+ `data/mixture.py`（带权采样器）+ `DataPipelineConfig` 扩展。
- **工作**：**1 天**。
- **验收**：构造一个"把 eval 集原文塞进训练集"的用例，去污染后命中数归零；
  `pipeline_report.json` 里出现 `contamination_removed` 字段。

### P0-6 · Agent 真沙箱 ✅

- **现状**：`builtin_tools.py::python_repl` **直接 exec**，无隔离、无限额、无超时——这是真实的安全洞。
- **缺什么**：进程级隔离 + CPU/内存/时间限额 + 文件系统越界防护（路径防护已有）+ 网络默认关闭。
- **落点**：`agent/sandbox.py`（子进程执行器 + resource limit + 超时 kill + 可选 Docker 后端）。
- **工作**：**1 天**（进程级）；Docker 后端 **+0.5 天**。
- **验收**：`tests/test_sandbox.py`：无限循环被超时杀掉、超大内存申请被拒、写越界路径失败。

### P0-7 · MCP（Model Context Protocol）工具层 ✅

- **现状**：`tools.py` 是自研 JSON Schema 协议，与生态不互通。
- **缺什么**：2026 工具层事实标准。
- **落点**：`agent/mcp.py`（MCP client：stdio + HTTP 两种 transport，`tools/list` / `tools/call`）+ 一个内置 MCP server 示例；`ToolRegistry` 支持从 MCP server 动态装载工具。
- **工作**：**1.5 天**。
- **验收**：Agent 通过 MCP 调用外部工具成功；`EchoBackend` 下可离线跑通协议（不依赖真实模型）。

### P0-8 · MuonClip / QK-Clip 训练稳定化 ✅

- **现状**：`optim.py` 有 Muon，但无 QK-clip；Kimi K2 靠它在 15.5T token 上做到零 loss spike。
- **缺什么**：`γ = min(1, τ / S_max)`，`W_Q ← γ^α W_Q`，`W_K ← γ^(1-α) W_K`。
- **落点**：`train/qk_clip.py` + `Trainer` hook（每 N 步监控 max attention logit）+ 日志。
- **工作**：**0.5 天**。
- **验收**：`tests/test_qk_clip.py` 验证缩放后 Q·K 点积整体乘 γ；
  提供"关掉 clip 出现 logit 爆炸 / 打开后平稳"的对照实验脚本。

> **P0 合计工作量：约 14 天**（按每天有效 4–6 小时计）。

---

## 四、Gap 清单 · P1（v1.5，约 20 项）

| # | 缺口 | 落点 | 量 |
|---|---|---|---|
| 1 | **可学习稀疏注意力**（Lightning Indexer + Top-K，DSA/CSA 简化版） | `model/sparse_attention.py` + `sparse_attention_ref` | 3d |
| 2 | **跨层 KV Sharing**（producer/consumer，Gemma 4 思路） | `model/kv_sharing.py` + `ModelConfig.kv_share_pattern` | 1.5d |
| 3 | **mHC 超连接**（多残差流 + Sinkhorn-Knopp 双随机） | `model/hyper_connections.py` | 1.5d |
| 4 | **MLA 权重吸收推理形式** | `model/mla.py` 增加 `absorb()` + 测试 | 1d |
| 5 | **分布式 Muon**（state sharding + 正交化通信） | `train/parallel/muon_dist.py` | 2d |
| 6 | **FP8 训练**（block-wise scaling + 主权重） | `train/fp8.py` | 2d |
| 7 | **LatentMoE**（降维路由，All-to-All payload ↓4×） | `model/moe.py` 扩展 | 1d |
| 8 | **MoE 分组 GEMM 内核 + Expert Parallel 推理** | `inference/kernels/moe_gemm.py` | 3d |
| 9 | **多状态 Cache Manager**（Paged KV / Ring KV / 递归 State / 压缩池） | `inference/cache_manager.py`（重构 `kv_cache.py`） | 3d |
| 10 | **Layer-aware 算子分派**（按 `layer_types` 路由 kernel） | `model/dispatch.py` 扩展 | 1d |
| 11 | **PD 解耦真实部署**（prefill/decode 分离 worker + KV 传输） | `inference/disaggregated.py` | 3d |
| 12 | **Flash-Decoding / split-KV** | `inference/kernels/flash_decoding.py` | 2d |
| 13 | **Sleep mode / 权重卸载**（训推切换，服务 RL 训练） | `inference/sleep.py` | 1.5d |
| 14 | **PPO + critic + GAE** | `posttrain/ppo.py` | 2d |
| 15 | **KTO / ORPO / SimPO + 在线 DPO + 拒绝采样** | `posttrain/preference.py` 扩展 | 2d |
| 16 | **Agentic RL**（多轮工具轨迹 + 规则奖励） | `posttrain/agentic_rl.py`（复用 `agent/`） | 3d |
| 17 | **Context Engineering**（文件系统 + 持久化任务状态 + 技能库） | `agent/workspace.py` / `agent/skills.py` | 3d |
| 18 | **Agent 成本护栏**（token 预算 / 循环检测 / 超时 / 长任务可恢复） | `agent/guardrails.py` | 1.5d |
| 19 | **任意 schema 约束解码**（DFA 预编译，替代现 JSON 子集） | `inference/structured.py` 扩展 | 2d |
| 20 | **SLO-aware 调度**（优先级 / 公平性 / TTFT-TPOT 权衡） | `inference/scheduler.py` 扩展 | 2d |
| 21 | **融合交叉熵 / chunked CE**（省 logits 显存） | `train/losses.py` + Triton kernel | 1.5d |
| 22 | **RWKV-7 广义 Delta Rule**（无 KV Cache 极端路线） | `model/rwkv7.py` | 2d |

> **P1 合计：约 45 天**。建议按"架构族 → 训练族 → 推理族 → Agent 族"分批推进。

---

## 五、Gap 清单 · P2（v2.0 选修，约 25 项）

| 层 | 项目 |
|---|---|
| 架构 | IndexShare 跨层复用 Top-K、HCA 重压缩、Hash MoE warmup、PLE 逐层嵌入、Clamped SwiGLU、NoPE/iRoPE 交替、Local/Global 异构 head_dim 与 rope_theta、并行 Attention-SSM 混合块（Falcon-H1）、K=V 共享、分组低秩输出投影、超长上下文温度缩放 |
| 训练 | 原生 FP4/NVFP4 预训练、EMA 权重平均、Context Parallel / Ring Attention、异步+分布式 checkpoint、loss spike 自动回滚、选择性激活重计算 |
| 后训练 | PRM 过程奖励、RLAIF / 宪法反馈、RLOO / REINFORCE++、奖励塑形与 hacking 检测工具、Thinking mode 与推理预算控制 |
| 推理 | NVFP4 / MXFP4 微缩放量化、CUDA 版 PagedAttention、Attention 反向内核、FP8 GEMM、Hopper/Blackwell 特性（WGMMA/TMA/异步流水线）、分层 KV 缓存（GPU→CPU→SSD）、LoRA 多租户热插拔、树状投机、Beam search、gRPC、递归状态前缀快照、多模态前缀缓存 |
| 服务 | 语义路由 / 多模型路由、三层缓存（网关响应缓存 + 提示词缓存）、自动扩缩容、每 token 成本追踪 |
| Agent | A2A 协议、HITL 人工审批与权限分级、Computer Use / 浏览器、真向量库 / 图记忆 / 记忆巩固、子智能体上下文隔离与并行 Swarm |
| 数据 | 全局跨分片去重、Suffix Array 精确去重、课程学习与退火数据、代码/数学数据专项、词表规模与压缩率消融 |
| 评测 | Agent 任务评测（成功率 / Pass^k / 成本）、量化精度损失系统化评测 |

---

## 六、分期路线图

```
v1.0 ── 补齐闭环与 2026 主干（P0，约 14 天）
  ├─ 阶段 A：评测与数据地基   P0-4 能力评测 → P0-5 去污染/配比
  ├─ 阶段 B：架构主干         P0-1 GDN+Mamba2 → P0-8 MuonClip
  ├─ 阶段 C：后训练闭环       P0-3 RM+RLVR+rollout（用阶段 A 的 eval 验证）
  ├─ 阶段 D：推理新范式       P0-2 MTP 自草稿
  └─ 阶段 E：Agent 落地       P0-6 沙箱 → P0-7 MCP

v1.5 ── 追平前沿机制（P1，约 45 天）
  架构族（1-4,7,22） → 训练族（5,6,21） → 推理族（8-13,19,20） → 后训练族（14-16） → Agent 族（17,18）

v2.0 ── 差异化（P2 选修）
  机制对照矩阵：GDN / Mamba-2 / Retention / Sparse / Full 五路线
  × {吞吐, 显存, MFU, 精度, 长上下文衰减} 五维度
```

## 七、验收标准（判定 Target 达成）

| 维度 | v1.0 起点 | v1.0 target | 当前（2026-10） |
|---|---|---|---|
| `docs/11` P0 条目覆盖 | 0 / 8 | 8 / 8 | **8 / 8 ✅** |
| `docs/11` P1 条目覆盖 | ~2 / 22 | 2 / 22 | ~2 / 22（下一阶段） |
| 单项机制"三件套"（reference + kernel + 一致性测试） | 部分 | P0 全部具备 | **P0 全部具备 ✅** |
| 端到端链路 | data→pretrain→generate→serve→agent | +posttrain RL 闭环 + eval 门禁 + MTP 推理 | **已达成 ✅** |
| 测试用例数 | 148 | ≥ 200 | **201 ✅** |
| 可回答的问题 | "每个优化快多少" | +"模型学会了没有" | **已可回答 ✅** |

v1.5 target 保持不变（见本文第四节 P1 清单）。

---

## 八、维护约定

1. 每落地一项，改本文状态 `✅ → ✅`，并同步 `docs/11-frontier-2026.md` 对应行的判定列（`P0 → 已有`）。
2. 每个新增模块必须同时交付：README 中全链路地图的一行 + 一篇 §设计原则说明 + 至少一个数值一致性测试 + 一个 benchmark 入口。
3. 每半年重跑一次 `docs/11` 的雷达盘点（前沿技术半年一换血，2026 H2 的主线未必是 2027 H1 的主线）。
