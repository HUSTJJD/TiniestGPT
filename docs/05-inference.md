# 推理优化与 AI Infra（本项目重点）

> 配套的 vLLM 对照（本项目模块 ↔ vLLM 文件、以及我们省略了什么）
> 见 [`07-vllm.md`](07-vllm.md)。建议两篇一起读。

> 推理系统的第一性问题：**decode 是 memory-bound**。
> 每生成一个 token 都要把「模型权重 + 全部 KV」从 HBM 读一遍，而计算量极小。
> 因此优化主线只有两条：**减少要读的数据**（GQA / MLA / 量化 / 投机解码）
> 与**让更多请求共享一次读取**（批处理 / 内核融合 / CUDA Graph）。

## 0. 先建立量化直觉

```
KV 显存/token = 2 · layers · n_kv_heads · head_dim · dtype_bytes
decode 单步时间 ≈ max( 权重带宽时间 , KV 带宽时间 )
吞吐          ≈ batch_size / 单步时间       ← batch 越大，权重读取被摊得越薄
```

`python -m tiniestgpt.cli info` 会打印每个预设的 `kv/token`，先记住这个数。

## 1. 优化层级与本项目的对应实现

| 层级 | 技术 | 代码 | 主要收益 |
|---|---|---|---|
| L0 | 无 KV Cache（每步重算全序列） | `benchmarks/inference_ablation.py` | 基线（O(L²)） |
| L1 | 稠密 KV Cache | `model/kv_cache.py::DenseKVCache` | O(L²)→O(L) |
| L2 | **PagedAttention + 连续批处理** | `scheduler.py` + `engine.py` | 显存几乎零碎片、吞吐 2~10× |
| L3 | **Prefix Caching** | `scheduler.py::PrefixCache` | 相同前缀零计算 |
| L4 | **KV Cache INT8 量化** | `quantization/kvquant.py` | 带宽减半 |
| L5 | **权重量化**（RTN / GPTQ / AWQ / SmoothQuant / NF4 / FP8） | `quantization/` | decode 带宽减半（W4A16） |
| L6 | **Triton 内核**（online softmax decode） | `kernels/triton_kernels.py` | 显存 O(L)→O(1)，省 launch |
| L7 | **投机解码** | `speculative.py` | 串行变并行，1.5~3× |
| L8 | **CUDA Graph** | `cuda_graph.py` | CPU launch 开销 O(N)→O(1) |
| — | 采样与解码策略 | `sampler.py` | 质量/多样性控制 |
| — | 服务化 | `server.py` | OpenAI 兼容生态 |

跑一次就全看到：

```bash
python benchmarks/inference_ablation.py --checkpoint out/tiny/last.pt --batch 8
```

## 2. PagedAttention：为什么必须有

稠密 KV Cache 的三个问题：

1. 必须按 `max_seq_len` 预分配 —— 实际利用率常常 < 50%；
2. 序列结束后显存不能给别人用 —— 碎片；
3. beam search / 共享前缀无法复用。

分页之后：

```
k_cache[layer][block][slot][head][dim]        # 物理块池
block_table[seq][i] = 第 i 个逻辑块 → 物理块号  # 逻辑→物理映射
```

收益：**按需分配**（几乎零浪费）、天然支持 prefix caching、支持抢占与重算。

> 实现细节：gather 出 int8 / fp16 的 KV 之后**再反量化**（`paged_attention_quantized_ref`），
> 否则就退化成"先全量反量化"，白白浪费省下的带宽。

## 3. 连续批处理：吞吐的真正来源

静态 batch 下，短请求必须等最长请求 → GPU 空转（尾延迟的来源）。
Continuous Batching 每一步都重新组 batch：完成的立刻离开、等待的立刻补位。

`Scheduler.schedule()` 的决策顺序：

1. **decode 优先**（保证正在生成的请求低延迟）；
2. 用剩余 token 预算接纳 waiting 请求；
3. 块不足时**抢占**（recompute 策略：释放块并退回等待队列）。

配套：**Chunked Prefill** —— 把超长 prompt 切成小块，避免一次 prefill 独占几百毫秒。

## 4. Prefix Caching

系统提示词 / 多轮对话前缀往往完全相同。按块哈希
`h_i = H(h_{i-1}, tokens_i)`，只缓存**完整块**（最后一个不满的块不缓存）。
命中时连计算都省了 —— 引擎只需 `computed` 之后的 token 做前向。

诊断：`engine.stats()["prefix_cache_hit_rate"]`。

## 5. 量化

### 权重量化

| 方案 | 粒度 | 特点 |
|---|---|---|
| RTN（round-to-nearest） | per-channel / per-group | 基线，一分钟搞定 |
| **GPTQ** | per-group(128) | 用 Hessian 逆把误差分摊到未量化的列（OBS 思想） |
| **AWQ** | per-input-channel 缩放 | 保住"激活大"的重要通道，缩放折进前一个 norm，零额外计算 |
| **SmoothQuant** | W8A8 | 把难度从激活迁移到权重（`s = max\|X\|^α / max\|W\|^(1-α)`） |
| **NF4**（QLoRA） | 分块 + 双重量化 | 信息论最优 4bit 码本 |
| **FP8 (E4M3)** | per-tensor / per-token | 硬件支持好（H100 / 4090） |

**decode 阶段用 W4A16**（只量化权重）通常最优，因为瓶颈是权重带宽；
**prefill / 大 batch 用 W8A8**，因为此时是计算受限。

```bash
python -m tiniestgpt.cli quantize --checkpoint out/tiny/last.pt --method rtn  --bits 4
python -m tiniestgpt.cli quantize --checkpoint out/tiny/last.pt --method gptq --bits 4 --group-size 128
```

### KV Cache 量化

`QuantizedPagedKVCache` 直接替换 `PagedKVCache`，接口完全一致：

```python
from tiniestgpt.inference.quantization import QuantizedPagedKVCache
engine.cache = QuantizedPagedKVCache(...)
engine.scheduler.set_allocator(engine.cache.allocator)
```

粒度：`per_token`（每个 token 一个 scale，最鲁棒）与 `per_head`（scale 表更小）。

## 6. Triton 内核

`kernels/triton_kernels.py` 提供：
**PagedAttention decode 内核**（online softmax，寄存器里维护 running max/sum，
中间结果完全不写回，显存 O(L)→O(1)）、融合 RMSNorm、融合 SwiGLU。

设计原则：**每个 Triton 内核都有一个数值等价的 PyTorch 参考实现**
（`model/kernels.py`），由 `dispatch.py` 按可用性自动选择后端。
Windows / CPU / 无 Triton 时静默降级，功能完全不受影响。

## 7. 投机解码

"小模型猜、大模型一次性验证"。接受-拒绝采样
`accept with p = min(1, p_target(x)/p_draft(x))`；拒绝时从 `(p_target - p_draft)_+` 重采样。

**数学保证：输出分布与直接用 target 采样完全一致**（不是近似）。
`tests/test_inference.py::test_speculative_matches_target_distribution` 验证这一点。

关键实现细节（本项目踩过的坑）：
- 要得到"位置 n 的分布"，必须 forward 位置 n-1 的 token —— 差一位就全错；
- 验证时喂 `[最后一个已知 token, d_1..d_γ]` 共 γ+1 个，
  `logits[i]` 才是位置 `cached+i` 的分布；
- 一轮结束后把最后一个 token 同步写回两个模型的缓存，保证状态一致。

## 8. CUDA Graph

decode 每步几百个小 kernel，相当一部分时间花在 launch 与 CPU 调度上。
CUDA Graph 把一步录成有向图，一次 replay 提交全部 kernel。

三个硬性条件：
1. **形状固定** → 按 `(batch_size, 长度档位)` 分桶捕获；
2. **指针固定** → 预分配静态缓冲区，每步只 `copy_`（含 block_table / seq_lens / slot_ids）；
3. **不能有 host 同步** → 捕获期间不能出现 `.item()`，
   所以 KV 长度必须提前算好并以整数形式传入（`CacheView.max_len`）。

## 9. 采样（`sampler.py`）

处理器顺序：penalty → temperature → min-p → top-k/top-p → typical-p。

- **min-p**：`p ≥ min_p · p_max`，候选集大小自适应，比 top-p 更能兼顾确定性与多样性；
- **repetition / frequency / presence penalty**：压复读，注意保持 logit 原符号；
- 另提供 **beam search** 参考实现（逐 token 前向，慢但清晰）。

## 10. 服务化（`server.py`）

```bash
python -m tiniestgpt.cli serve --checkpoint out/tiny/last.pt --port 8000
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_tokens": 32}'
```

OpenAI 兼容（`/v1/completions`、`/v1/chat/completions` 含 SSE 流式、`/v1/models`、`/stats`），
生态里的客户端可以直接接进来。

## 11. 排错清单

| 症状 | 常见原因 |
|---|---|
| KV Cache 结果 ≠ 全序列前向 | 因果掩码没带 `offset`（SDPA 的 `is_causal` 不知道缓存偏移） |
| 长文本突然崩 | 窗口注意力没有配 Attention Sink |
| 量化后乱码 | 量化了 `lm_head` / MoE gate / embedding（默认已跳过） |
| 吞吐没提升 | batch 太小（权重带宽没被摊薄）或 prefill 占比过高 |
| CUDA Graph 捕获失败 | 捕获路径里有 `.item()` 或张量被重建（必须 `copy_`） |
