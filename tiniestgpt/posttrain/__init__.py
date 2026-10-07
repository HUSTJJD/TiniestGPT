"""后训练（Post-training）：让预训练模型"会说话、懂偏好、能推理"。

1. **SFT**：在 (instruction, response) 上做监督微调，**只对 response 部分算 loss**；
2. **DPO / IPO**：直接用偏好对优化策略，不需要显式奖励模型；
3. **GRPO**（DeepSeekMath / R1）：对同一个 prompt 采样一组答案，
   用**组内相对优势**替代 critic 网络，省掉一个和策略同规模的价值模型；
4. **蒸馏**：把大教师的 logit / 隐层知识压进小模型。
"""

from .sft import SFTConfig, build_sft_batch, sft_loss, SFTTrainer
from .dpo import DPOConfig, dpo_loss, sequence_logps
from .grpo import GRPOConfig, compute_group_advantages, grpo_loss
from .distill import kd_loss, hidden_state_loss

__all__ = [
    "SFTConfig", "build_sft_batch", "sft_loss", "SFTTrainer",
    "DPOConfig", "dpo_loss", "sequence_logps",
    "GRPOConfig", "compute_group_advantages", "grpo_loss",
    "kd_loss", "hidden_state_loss",
]
