# 模型架构：小参数量、全先进技术

默认配置约 25M 参数（`recipes/pretrain_tiny.yaml`），但结构对齐 2025-2026 的主流做法。

```
tok_emb → [ Block × L ] → norm_f → lm_head(与 embedding 共享权重)

Block:
  x ─► norm1 ─► Mixer ─► post_norm1 ─► (+) ─► norm2 ─► FFN/MoE ─► post_norm2 ─► (+)
```

## 1. 归一化

- **RMSNorm**：只除以均方根，不减均值、不要 bias。更稳、更快。统计量一律在 fp32 下算。
- **Sandwich（pre + post norm）**：Qwen3 / Gemma2 的做法，抑制深层 activation 数值爆炸。
  > ⚠️ 千万不要把 post-norm 的权重初始化为 0——那等价于"该层不参与"且梯度被截断。
  本项目把 `post_norm_init_zero` 默认设为 `False`，就是踩过这个坑。
- **QK-Norm**：对 Q/K 的 head 维做归一化，让 attention logits 的量级与层数解耦。
- **Dynamic Tanh (DyT)**：`tanh(αx)·w + b`，2025 年提出，直接挑战"必须重新中心化"的假设。

## 2. 位置编码（`rope.py`）

- **RoPE**：把每两个维度看成复数乘 `e^{i·m·θ}`，点积只依赖相对位置。
  `tests/test_model.py::test_rope_relative_property` 验证了这条性质。
- **YaRN**：低频插值、高频保留（ramp mask 平滑过渡）+ attention scale 修正
  `(0.1·ln s + 1)²`。用极少长文本数据即可扩上下文。
- **mRoPE**：不同维度段用不同位置索引（t/h/w），多模态标配。
- **部分旋转**：只让一部分维度旋转（MLA 常用），给"无位置信息的语义通道"留空间。

## 3. 注意力（`attention.py` / `mla.py`）

| 技术 | 解决什么 | 代价 |
|---|---|---|
| **GQA / MQA** | decode 的 KV 带宽；KV 头数 H→g，显存与带宽直接除以 H/g | 极小质量损失 |
| **滑动窗口** | 把长上下文注意力从 O(L²) 降到 O(L·W) | 长程依赖靠层间传递 |
| **Attention Sink** | 前几个 token 被所有位置强烈关注，挤出窗口会导致模型崩溃 | 必须常驻 |
| **logit soft-capping** | 压住极端 logits，比硬裁剪平滑，利于低精度稳定 | 一次 tanh |
| **MLA** | 把 KV 压成一条 latent，KV Cache 体积大幅下降 | Q/K 需拆成 nope+rope 两段 |

MLA 的形状细节（很容易写错）：
- `q = [q_nope(d_c), q_pe(d_r)]`，`k = [k_nope(d_c), k_pe(d_r)]`，**`v` 只有 `d_c`**；
- 因此 attention 输出的 head 维度是 `d_c`，`o_proj` 的输入是 `H·d_c`，不是 `H·(d_c+d_r)`；
- 缓存的是 `latent(d_latent) + 解耦 rope(d_r)`，每 token 一条。

## 4. FFN 与 MoE（`mlp.py` / `moe.py`）

- **SwiGLU**：`SiLU(xW_gate) ⊙ (xW_up)`，gate/up 融合成一次 GEMM（省一次 kernel launch）。
- **稀疏 MoE** 三件套：
  1. top-k 路由 + 细粒度专家 —— FLOPs 随 k 增长，参数量随专家数增长；
  2. **共享专家** —— 承载通用知识，明显更稳；
  3. **无辅助损失负载均衡**（DeepSeek V3）—— 给每个专家一个可动态更新的 bias，
     忙的降、闲的升，不干扰梯度。`moe_aux_coef=0` 时启用（默认），
     `>0` 时改用 Switch Transformer 的辅助损失。
- 实现要点：路由后按专家 **sort 一次**再连续切片，等价于分组 GEMM；
  容量限制用"专家内序号"向量化计算，避免 Python 循环。

诊断指标：`MoEStats.max_violation`（`|load-mean|/mean` 的最大值），越小越均衡。

## 5. 混合层（`linear_attention.py`）

`layer_types: "window,linear,window,full"` 可以按层指定类型。

`linear` 层是 RetNet 式 retention，有**两种数学等价的形态**：
- 并行形态（训练/prefill）：`(q·kᵀ) ⊙ decay_mask`，O(L²) 但可并行；
- 递推形态（decode）：`S_t = γ·S_{t-1} + k_t v_tᵀ`，每步 O(d²)，**不需要 KV Cache**。

`tests/test_model.py::test_linear_attention_parallel_equals_recurrent` 验证了二者等价——
这是"训练像 Transformer、推理像 RNN"的全部秘密。

## 6. 初始化

- 所有矩阵参数 `N(0, init_std)`；
- **depth-scaled init**：残差分支的输出投影（`o_proj` / `w_out`）按 `1/√(2L)` 缩小，
  防止 activation 随层数累积放大。

## 7. 动手实验清单

```bash
python -m tiniestgpt.cli info          # 打印所有预设的参数量与 kv/token
```

| 想理解的东西 | 改什么 | 看什么 |
|---|---|---|
| GQA 省了多少 | `attn_type: mha → gqa` | `kv/token`、训练吞吐 |
| MLA 省了多少 | `attn_type: mla` | `kv/token`、显存 |
| 窗口必须配 sink | `attn_sinks: 4 → 0` | 长文本生成是否崩溃 |
| MoE 的条件计算 | `moe_enabled: true` | `active params` vs `total params`、`max_violation` |
| 混合架构 | `layer_types` | 吞吐与长文本表现 |
