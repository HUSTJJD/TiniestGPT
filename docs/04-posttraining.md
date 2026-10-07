# 后训练：从"会续写"到"会帮忙"

预训练得到的是"下一个 token 预测器"；后训练把它变成"助手"。

## 1. SFT（`posttrain/sft.py`）

两个决定效果的细节：

- **prompt masking**：只对 response 算 loss。
  若把 prompt 也算进去，模型会去学"复述问题"，且短回答的梯度被稀释。
- **packing + 块对角掩码**：多条样本拼成定长序列提高吞吐，
  但必须配合 `make_doc_mask` 防止样本之间互相"泄题"。

```python
from tiniestgpt.posttrain import SFTConfig, SFTExample, build_sft_batch, sft_loss
batch = build_sft_batch([SFTExample(prompt="2+2=?", response="4")], tokenizer, SFTConfig())
loss = sft_loss(model(batch["input_ids"], attn_mask=batch["attn_mask"]), batch["labels"])
```

## 2. DPO / IPO（`posttrain/dpo.py`）

RLHF 的漂亮结论：最优策略与奖励的关系是
`r(x,y) = β·log(π(y|x)/π_ref(y|x)) + const`，
于是偏好概率可以写成只含策略的形式，**绕开显式奖励模型**：

```
L_DPO = -log σ( β·[(logπ(y_w)-logπ_ref(y_w)) - (logπ(y_l)-logπ_ref(y_l))] )
```

工程要点：
- 必须先算一份 **reference logps**（`no_grad` 前向）并缓存，否则每步跑两次模型；
- `β` 越大越保守（越贴近参考模型），典型 0.1；
- **IPO** 用平方损失替代 sigmoid，缓解 DPO 在弱偏好对上的过拟合。

## 3. GRPO（`posttrain/grpo.py`）

DeepSeek-R1 同族算法。与 PPO 的区别：**没有 critic**。

对同一 prompt 采样 G 条回答 → 奖励打分 → 组内标准化：
`Â_i = (r_i - mean(r)) / (std(r) + eps)`。
这一步"组内相对比较"就是优势估计，省掉一个与策略同规模的价值网络。

实现的进阶细节：
- PPO 的 clipped surrogate 防止单步更新过猛；
- **clip-higher**（DAPO）：上界比下界宽，鼓励探索低概率 token；
- **KL 惩罚** 支持 k1（有偏低方差）与 k3（无偏，恒正）两种估计器；
- **loss 粒度**：`token_mean`（默认）与 `seq_mean`（DAPO，避免长序列主导梯度）。

## 4. 蒸馏（`posttrain/distill.py`）

- **logit 蒸馏**：`T²·KL(student/T ‖ teacher/T)`。温度把概率摊平，
  暴露类别之间的相对关系（软标签比 one-hot 信息量大得多）。
- **隐层蒸馏**：用可学习的投影对齐中间表示再算 MSE / cosine，
  对"学生比教师小很多"的情况尤其有效。

## 5. 建议的实验顺序

1. 先 SFT，观察输出是否从"续写"变成"回答"；
2. 再 DPO，观察是否更符合偏好（用成对样本人工检查）；
3. 最后 GRPO，观察奖励曲线与 `clip_frac` / `kl` 是否正常
   （`clip_frac` 长期 > 0.3 说明单步更新过大，应减小 lr 或收紧 clip）。
