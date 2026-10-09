"""训练配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..model.config import ModelConfig

__all__ = ["TrainConfig"]


@dataclass
class TrainConfig:
    # ---------------- 数据 ----------------
    data_dir: str = "data/packed"
    batch_size: int = 8
    seq_len: int = 1024
    doc_mask: bool = False
    num_prefetch: int = 2

    # ---------------- 步数 ----------------
    max_steps: int = 3000
    grad_accum_steps: int = 4
    eval_every: int = 200
    eval_batches: int = 20
    log_every: int = 20
    save_every: int = 500

    # ---------------- 优化器 ----------------
    optimizer: str = "muon"            # adamw | muon | sophia | lion
    lr: float = 3e-4
    muon_lr: float = 0.02              # Muon 给矩阵参数的 lr（通常比 AdamW 大）
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.1
    no_decay_1d: bool = True           # bias / norm / embedding 不做 weight decay
    betas: Tuple[float, float] = (0.9, 0.95)
    momentum: float = 0.95
    ns_steps: int = 6                  # Muon 的 Newton-Schulz 迭代步数
    adamw_eps: float = 1e-8
    grad_clip: float = 1.0

    # ---------------- 学习率调度 ----------------
    lr_schedule: str = "wsd"           # wsd | cosine | linear | constant | trapezoid
    warmup_steps: int = 100
    decay_fraction: float = 0.2        # WSD 中 decay 阶段占比
    decay_type: str = "linear"         # WSD decay 的形状：linear | cosine | sqrt | 1-sqrt

    # ---------------- 精度与编译 ----------------
    dtype: str = "bf16"                # fp32 | bf16 | fp16
    compile_model: bool = False
    gradient_checkpointing: bool = False
    attn_backend: str = "auto"

    # ---------------- 损失 ----------------
    label_smoothing: float = 0.0
    z_loss_weight: float = 0.0
    moe_aux_weight: float = 0.01
    mtp_loss_weight: float = 0.0       # MTP 辅助损失权重（需 model.mtp_enabled=True）

    # ---------------- 训练稳定化（2026） ----------------
    # MuonClip / QK-Clip：约束 attention logit 的最大值，防止 softmax 饱和与 loss spike。
    # Kimi K2 靠它在 15.5T token 上做到零 spike。0 = 关闭。
    qk_clip_tau: float = 0.0
    qk_clip_every: int = 50           # 每多少步做一次 clip
    qk_clip_alpha: float = 0.5        # Q/K 各承担的缩放比例（0.5 = 各担 sqrt）

    # ---------------- 分布式 ----------------
    distributed: str = "none"          # none | ddp | fsdp
    fsdp_sharding: str = "full"        # full | hybrid
    bucket_cap_mb: int = 25            # DDP gradient bucket 大小（影响通信重叠）

    # ---------------- 其它 ----------------
    out_dir: str = "out/tiny"
    seed: int = 42
    resume: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)

    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum_steps
