# 预训练工程：把算力变成 loss

## 1. 一次 step 在做什么（`train/engine.py`）

```
for step:
    for i in range(grad_accum_steps):        # 梯度累积：用小显存模拟大 batch
        loss = CE(logits, labels) / grad_accum
        loss.backward()                      # autocast(bf16) 下自动管理精度
    unscale → clip_grad_norm → 检查 NaN/Inf → optimizer.step → scheduler.step
```

刻意保留的**工程细节**（每一条都是真实训练里会踩的坑）：

- **bf16 autocast + fp32 主权重**：参数始终是 fp32，只在计算时降精度。
  bf16 的动态范围与 fp32 一致，因此**不需要 GradScaler**；fp16 才需要。
- **裁剪前必须 `unscale_`**：否则裁剪阈值被 scale 放大，等于没裁剪。
- **NaN/Inf 守卫**：出现坏梯度时跳过这一步，而不是让模型被污染。
- **吞吐与 MFU**：`MFU = 实际 FLOPs/s ÷ 硬件峰值`。
  只有把 FLOPs 算清楚（`transformer.flops_per_token()`），才知道优化有没有用。

## 2. 优化器（`train/optim.py`）

| 优化器 | 思想 | 适用 |
|---|---|---|
| AdamW | 一阶动量 + 二阶自适应 + 解耦 weight decay | 基线，最稳 |
| **Muon** | 对矩阵参数做 **Newton-Schulz 正交化**，让每步在所有方向上均匀前进 | 小模型/小 batch 上收敛更快 |
| Sophia-G | Hutchinson 估计 Hessian 对角线，按曲率裁剪 | 对 loss 尖刺更鲁棒 |
| Lion | 只用 sign，动量只需一份（省一半优化器显存） | 显存紧张时 |

Muon 的关键：把更新方向变成近似半正交矩阵（`X Xᵀ ≈ I`），
避免被少数大奇异值方向主导。实现上用五次多项式迭代拟合 `x^{1/2}`，避免 SVD。
实践用法：**矩阵参数用 Muon，embedding / bias / norm 用 AdamW**（本项目已内置该分组）。

## 3. 学习率调度（`train/lr_sched.py`）

**WSD（Warmup-Stable-Decay）** 是近年大模型预训练的事实标准：

- warmup：线性升到峰值；
- stable：**保持常数** —— 这意味着你不必事先承诺训练多少步；
- decay：**在任意时刻** `scheduler.start_decay()` 就能开始退火，
  且退火中途的 checkpoint 也是可用的，可以做"中途退火 + 早停"的实验循环。

对比 cosine：必须提前定死总步数，中途改计划就要重训。

decay 形状：`linear` / `cosine` / `sqrt` / `1-sqrt`（默认，前期掉得快）。

## 4. 损失函数（`train/losses.py`）

- **交叉熵**：唯一的主损失；
- **label smoothing**：预训练一般不开（会损失 log-likelihood），SFT 时可小量使用；
- **z-loss**：`λ·mean(logsumexp(logits)²)`，惩罚 logits 的整体平移，
  显著减少训练中的 loss 尖刺与 bf16 数值问题；
- **MoE 辅助损失**：由 `model.last_aux_loss` 传回并加权。

## 5. 分布式（`train/distributed.py`）

- **DDP**：每卡一份完整参数，反向时 all-reduce 梯度。
  优化点：`gradient_as_bucket_view`（省一次拷贝）+ 大 bucket + `static_graph=True`
  （结构固定时省掉每步重建 bucket）→ 让通信与计算重叠。
- **FSDP**：参数/梯度/优化器状态分片，前向按需 all-gather、用完即释放。
  显存近乎除以 N，代价是额外通信。`ModuleWrapPolicy({TransformerBlock})` 按层分片。

选择依据：单卡能放下 → DDP；放不下 → FSDP。

## 6. Checkpoint（`train/checkpoint.py`）

- 只保存 rank0（或 FSDP 的 full state dict）；
- **异步保存**：后台线程 + CPU 快照，把写盘从关键路径移除；
- optimizer 状态体积≈参数量（Muon 还有动量缓冲），磁盘紧张时可 `save_optimizer=False`。

## 7. 动手实验

```bash
# Muon vs AdamW
python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml --set optimizer=adamw
python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml --set optimizer=muon

# 梯度检查点的代价
python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml --set gradient_checkpointing=true

# 两卡 DDP
torchrun --nproc_per_node=2 -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml \
    --set distributed=ddp
```

观察：loss 曲线、tokens/s、MFU、显存峰值四项。
