# 09 · 分布式训练：显存账本与 3D 并行

> 对应 AIInfraGuide 路线**第二层**。
> 代码在 `tiniestgpt/train/parallel/` 与 `tiniestgpt/common/memory_ledger.py`。

---

## 0. 一句话世界观

> 所有的并行策略都是在回答同一个问题：
> **模型状态（参数 / 梯度 / 优化器）和激活，到底怎么分给 N 张卡？**

| 策略 | 切什么 | 通信 | 通常部署在哪里 |
|---|---|---|---|
| DP / DDP | 不切状态，只切数据 | 反向时 1 次 all-reduce 梯度 | 任意（可跨机） |
| **ZeRO-1/2/3** | 优化器状态 → +梯度 → +参数 | all-reduce / reduce-scatter / all-gather | 任意（可跨机） |
| **TP** 张量并行 | 矩阵乘的**维度** | 每层 2 次 all-reduce | **只能机内**（NVLink） |
| **SP** 序列并行 | 序列维（TP 区之外） | all-gather + reduce-scatter | 机内（配合 TP） |
| **PP** 流水线并行 | **层** | 相邻 stage 间 send/recv 激活 | 跨机（通信量小） |
| EP 专家并行 | MoE 的专家 | all-to-all | 跨机 |

---

## 1. 显存账本（先算账，再动手）

`common/memory_ledger.py`。核心公式（字节）：

```
参数       = 2 · N          （bf16/fp16 计算副本）
梯度       = 2 · N
AdamW 状态 = 12 · N        （fp32 主权重 4N + 一阶动量 4N + 二阶动量 4N）
激活       = L · B · S · bytes_per_token
KV Cache   = 2 · L · Hkv · D · S · B · dtype
```

**关键直觉**：AdamW 下优化器状态（84GB @ 7B）是参数本身（14GB）的 **6 倍**，
所以 ZeRO 第一刀必须砍在优化器状态上——这就是 ZeRO-1 存在的原因。

```python
from tiniestgpt.common.memory_ledger import ModelShape, estimate_training_memory, format_report

shape = ModelShape(n_params=7_000_000_000, n_layers=32, dim=4096, hidden_dim=11008,
                   n_heads=32, n_kv_heads=32, head_dim=128, vocab_size=32000,
                   seq_len=2048, batch_size=8)
print(format_report(estimate_training_memory(shape, "adamw", world_size=8, zero_stage=3)))
```

几个可以直接口算的结论（都有测试兜底）：

* 7B + AdamW = **112 GB** 状态（16 字节/参数），单卡 80GB 放不下 → 必须上 ZeRO
* LLaMA-2-7B，`S=4096, B=16, fp16` → KV Cache ≈ **32 GiB**
* GQA 把 `Hkv` 从 32 降到 8 → KV Cache 直接 **1/4**
* 重计算把激活从"每层全存"降到"只存每层输入" → 激活显存降一个数量级

---

## 2. ZeRO：用通信换显存（`parallel/zero.py`）

| 阶段 | 切分对象 | 反向时的梯度处理 | 每卡状态（N=8, 7B） |
|---|---|---|---|
| 0（DDP） | 无 | all-reduce | 112 GB |
| **1** | 优化器状态 | all-reduce | 24.5 GB |
| **2** | + 梯度 | reduce-scatter | 15.75 GB |
| **3** | + 参数 | reduce-scatter | 14 GB |

一句话讲清 2 和 3 的差别：

> ZeRO-2 只在 backward 时按需 AllReduce 梯度，参数每卡各存一份；
> ZeRO-3 连参数也切了，forward/backward 都要 all-gather 拿参数、用完即弃，
> 通信量约翻倍但每卡显存降到 1/N。

```python
from tiniestgpt.train.parallel import DistContext, ZeroOptimizer

zero = ZeroOptimizer(model.parameters(), DistContext(world_size=1), stage=2, lr=1e-3)
loss.backward()
zero.step()
zero.zero_grad()
```

`world_size=1` 时与 `torch.optim.AdamW` **数值完全等价**（有回归测试）。

---

## 3. 张量并行（`parallel/tp.py`）

```
        ┌── ColumnParallel（按 output 切，无通信）──┐
  x ────┤                                          ├── RowParallel（按 input 切）── all_reduce ── y
        └── ColumnParallel ────────────────────────┘
              Q/K/V 投影、FFN 的 gate/up              O 投影、FFN 的 down 投影
```

* `q/k/v_proj`、`mlp.w_in` → `ColumnParallelLinear`（输出是分片，正好对应"本卡负责几个头"）
* `o_proj`、`mlp.w_out` → `RowParallelLinear`（输入是分片，**末尾必须 all-reduce**）
* bias 必须放在 all-reduce **之后**加，否则会被 `world_size` 放大

```python
from tiniestgpt.train.parallel import DistContext, apply_tensor_parallel

ctx = DistContext(world_size=2, rank=rank)
apply_tensor_parallel(model, ctx, rank=rank)     # 自动改 q/k/v/o_proj 与 FFN，并同步缩小头数
```

**为什么 TP 不能跨机**：每个 forward+backward 要做 2 次 all-reduce，
通信量 ∝ `B·S·d`，只有 NVLink（900 GB/s）扛得住；IB（200 Gb/s ≈ 25 GB/s）会直接把算力饿死。

---

## 4. 序列并行（`parallel/sp.py`）

TP 只对矩阵乘有收益，LayerNorm / Dropout / 残差在 TP 下**每卡各算一份**，
激活显存一点没省。SP 把这些区域改成沿序列维切分：

```
[B,T,C] --reduce-scatter--> [B,T/N,C] --all-gather--> [B,T,C] --(TP 区)--reduce-scatter--> ...
           LayerNorm/Dropout             attention / FFN
```

通信量（2·B·T·C）与 TP 区内的 2 次 all-reduce 同量级，
所以 SP 是"**用同样的通信量换激活显存**"——几乎白赚。

---

## 5. 流水线并行（`parallel/pp.py`）

把层切给不同设备，用**微批次**填满气泡：

```
bubble_ratio ≈ (num_stages - 1) / num_microbatches
```

**1F1B** 的标准调度：warmup 灌入若干微批次后，每个 step 严格"一个 forward + 一个 backward"，
把激活显存从 `O(num_microbatches)` 压到 `O(num_stages)`——这是它能跑千亿模型的关键。

```python
from tiniestgpt.train.parallel import PipelineParallel

pipe = PipelineParallel(model.layers, num_stages=4, num_microbatches=8)
```

---

## 6. 3D 并行怎么配（64 卡 = 8 节点 × 8 卡）

```
TP = 8   （机内，走 NVLink）
PP = 4   （跨机，通信量最小）
DP = 2   （剩余维度）
```

配置顺序的心法：**通信最频繁的放带宽最高的地方**。

| 维度 | 通信频率 | 放哪 |
|---|---|---|
| TP | 每层 2 次 all-reduce | 机内 NVLink |
| SP | 每段 2 次 | 跟 TP 同组 |
| PP | 每个微批次 1 次点对点 | 跨机 IB |
| DP / ZeRO | 每步 1 次 | 跨机 IB（可与计算重叠） |

---

## 7. 没有多卡怎么学：`DistContext` 的模拟模式

`parallel/comm.py` 提供了三种模式：

| 情况 | 行为 |
|---|---|
| `world_size=1` | 所有集合通信是恒等变换，并行模块退化为普通实现 |
| 起了 `torch.distributed` | 走真正的 NCCL / gloo |
| `world_size>1` 但没起进程组 | **单进程依次扮演每张卡**（`run_ranks`），集合通信用累加/拼接算出真实结果 |

所以下面这段代码在**笔记本上**就能验证"两张卡的 TP 结果 == 单卡结果"：

```python
from tiniestgpt.train.parallel import DistContext, run_ranks
from tiniestgpt.train.parallel.tp import ColumnParallelLinear

ctx = DistContext(world_size=2)
out = run_ranks(ctx, lambda r: ColumnParallelLinear.from_linear(lin, ctx, r, gather_output=True)(x))
assert torch.allclose(out, lin(x))
```

注意：模拟模式下只有**最后一个 rank** 的返回值是完全正确的（前面的 rank 只保证形状正确），
所以 `run_ranks` 只返回最后一次的结果。

---

## 8. 检验标准

- [ ] 拿到 7B 配置能口算：fp16 参数 14GB、Adam 状态 56~84GB，判断 80GB 单卡够不够
- [ ] 讲清 ZeRO-2 与 ZeRO-3 的差别（参数是否也切、通信量是否翻倍）
- [ ] 30 分钟内把单卡脚本改成 DDP（`train/distributed.py` 已封装）
- [ ] 给 64 卡集群设计 TP=8 / PP=4 / DP=2 的拓扑，并说明 TP 为什么不能跨机
- [ ] 解释 BF16 为什么比 FP16 更适合训练（指数位 8 vs 5，动态范围接近 FP32，多数情况不用 loss scaling）

---

## 9. 真跑起来

```bash
torchrun --standalone --nproc_per_node=2 -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml
```

> Windows 不支持 NCCL，多卡实验请在 Linux / WSL2 上做；
> 单卡下用上面第 7 节的模拟模式验证数学正确性即可。
