"""模型配置：一个 dataclass 就是一份"架构设计说明书"。

建议你逐项开关这些字段并观察 loss / 吞吐 / 显存的变化——这比读十篇论文直观。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

__all__ = ["ModelConfig", "PRESETS"]


@dataclass
class ModelConfig:
    # ---------------- 基础骨架 ----------------
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2            # GQA：K/V 头数 < Q 头数
    head_dim: int = 0              # 0 → dim // n_heads
    hidden_dim: int = 0            # 0 → 按 SwiGLU 经验公式自动推导
    max_seq_len: int = 2048
    multiple_of: int = 64

    # ---------------- 归一化 ----------------
    norm_type: str = "rms"         # rms | layernorm | dynamic_tanh
    norm_eps: float = 1e-6
    pre_norm: bool = True          # 残差前的 norm（现代标配）
    post_norm: bool = True         # 残差后的 norm（Sandwich / Qwen3 / Gemma2）
    # 警告：置为 True 会把残差分支整体乘以 0，等价于"该层不参与"，且梯度被截断。
    # 仅用于消融实验；正常训练必须为 False。
    post_norm_init_zero: bool = False
    qk_norm: bool = True           # QK-Norm：抑制 attention logits 爆炸（scale 不稳定时救命）

    # ---------------- 位置编码 ----------------
    rope_type: str = "yarn"        # rope | yarn | mrope | none
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0      # YaRN 的上下文扩展倍数
    rope_original_max_len: int = 2048
    rope_partial: float = 1.0      # 参与旋转的维度比例（MLA 下常取 0.25~0.5）
    rope_mscale: float = 1.0       # 额外的 attention scale 修正（YaRN 用）

    # ---------------- 注意力 ----------------
    attn_type: str = "gqa"         # gqa | mha | mqa | mla
    attn_backend: str = "auto"     # auto | ref | sdpa | triton
    attn_window: int = -1          # 滑动窗口大小；-1 = 全注意力
    attn_sinks: int = 4            # Attention Sink：永远可见的前 k 个 token（StreamingLLM）
    attn_softcap: float = 0.0      # logit soft-capping（Gemma2）；0 = 关闭
    mla_latent_dim: int = 64       # MLA 的 KV 压缩维度
    mla_rope_dim: int = 32         # MLA 解耦 RoPE 的维度

    # ---------------- FFN ----------------
    act_type: str = "swiglu"       # swiglu | geglu | gelu | relu2
    dropout: float = 0.0

    # ---------------- 稀疏 MoE ----------------
    moe_enabled: bool = False
    n_experts: int = 8
    n_experts_per_tok: int = 2
    n_shared_experts: int = 1      # 共享专家（DeepSeek）
    moe_score_func: str = "sigmoid"  # sigmoid(DeepSeek V3) | softmax(Switch)
    moe_aux_coef: float = 0.0      # >0 走辅助损失；=0 走 bias 动态均衡（无损失、更稳）
    moe_bias_lr: float = 0.001
    moe_capacity_factor: float = 0.0   # >0 启用容量限制（token drop）

    # ---------------- 参数与初始化 ----------------
    tie_embeddings: bool = True
    init_std: float = 0.02
    depth_scaled_init: bool = True  # 残差分支按 1/sqrt(2L) 缩放（GPT-2 式 depth scaling）
    z_loss_weight: float = 0.0      # z-loss：压住 logits 的 log-sum-exp 漂移

    # ---------------- 层类型 ----------------
    # 逗号分隔：full | window | linear | gdn | mamba2。留空 = 全部 full。
    # 例："window,window,full" 会循环应用到各层（滑动窗口与全注意力交替，兼顾效率与长程）。
    #     "gdn,gdn,gdn,full" 是 Qwen3.6 式的 3:1 混合（线性记忆 + 周期全局校正）。
    layer_types: str = ""

    # ---------------- 线性 / SSM 序列混合器（2026） ----------------
    # layer_types 里出现 "linear" 时使用哪种线性注意力：
    #   retention | gdn(Gated DeltaNet, Qwen3.6/Kimi Linear) | mamba2(SSD, Nemotron/Falcon)
    seq_mixer: str = "retention"
    gdn_chunk_size: int = 64        # GDN 的分块大小；<=0 走朴素逐步递推（参考实现）
    gdn_conv_kernel: int = 0        # >0 时对 q/k/v 加深度可分离卷积（Qwen3.6 的做法）
    ssm_state_size: int = 32        # Mamba-2/SSD 的状态维度 N
    ssm_conv_kernel: int = 4        # Mamba-2 的短卷积宽度
    ssm_n_heads: int = 0            # 0 → 用 n_heads
    ssm_chunk_size: int = 64        # SSD 的分块大小

    # ---------------- 训练相关 ----------------
    gradient_checkpointing: bool = False

    # ---------------- MTP 多 token 预测（2026 标配） ----------------
    mtp_enabled: bool = False
    mtp_n_predict: int = 1          # 额外预测几个未来 token（DeepSeek 用 1，MiniMax 用 3）

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            self.head_dim = self.dim // self.n_heads
        if self.attn_type == "mha":
            self.n_kv_heads = self.n_heads
        elif self.attn_type == "mqa":
            self.n_kv_heads = 1
        elif self.attn_type == "mla":
            self.n_kv_heads = 1          # MLA 只缓存一条 latent
        if self.n_kv_heads <= 0:
            self.n_kv_heads = 1
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads({self.n_heads}) 必须能被 n_kv_heads({self.n_kv_heads}) 整除")
        if self.hidden_dim <= 0:
            # SwiGLU 的经验公式（Llama）：让参数量与"普通 4d FFN"对齐
            raw = int(2 * (4 * self.dim) / 3)
            self.hidden_dim = int(self.multiple_of * math.ceil(raw / self.multiple_of))
        if self.vocab_size % 64 != 0:
            self.vocab_size = int(64 * math.ceil(self.vocab_size / 64))

    # ------------------------------------------------------------------ #
    @property
    def n_groups(self) -> int:
        """GQA 的分组数（每个 KV 头服务多少个 Q 头）。"""
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def layer_type_list(self) -> List[str]:
        """把 layer_types 展开成逐层的类型列表。"""
        if not self.layer_types:
            base = "window" if self.attn_window > 0 else "full"
            return [base] * self.n_layers
        types = [t.strip() for t in self.layer_types.split(",") if t.strip()]
        return [types[i % len(types)] for i in range(self.n_layers)]

    _LAYER_TYPES = ("full", "window", "linear", "gdn", "mamba2")

    def validate(self) -> None:
        for t in set(self.layer_type_list()):
            if t not in self._LAYER_TYPES:
                raise ValueError(f"未知层类型: {t}（可选 {'/'.join(self._LAYER_TYPES)}）")
        if self.attn_type not in ("gqa", "mha", "mqa", "mla"):
            raise ValueError(f"未知注意力类型: {self.attn_type}")
        if self.seq_mixer not in ("retention", "gdn", "mamba2"):
            raise ValueError(f"未知线性混合器: {self.seq_mixer}（可选 retention/gdn/mamba2）")


# --------------------------------------------------------------------------- #
# 预设
# --------------------------------------------------------------------------- #
def _preset(**kw) -> ModelConfig:
    return ModelConfig(**kw)


PRESETS = {
    # ~12M：CPU 上也能跑
    "nano": _preset(dim=256, n_layers=8, n_heads=8, n_kv_heads=2, hidden_dim=768,
                    vocab_size=4096, attn_window=512),
    # ~25M：默认，单卡 3060 上数分钟见效
    "tiny": _preset(dim=384, n_layers=12, n_heads=8, n_kv_heads=2, hidden_dim=1024,
                    vocab_size=4096, attn_window=1024),
    # ~60M：GQA + 滑动窗口 + sink，验证长上下文
    "small": _preset(dim=512, n_layers=16, n_heads=8, n_kv_heads=2, hidden_dim=1408,
                     vocab_size=8192, attn_window=1024, attn_sinks=4),
    # ~110M（激活 ~40M）：稀疏 MoE 版，验证路由与负载均衡
    "moe": _preset(dim=512, n_layers=12, n_heads=8, n_kv_heads=2, hidden_dim=1024,
                   vocab_size=8192, moe_enabled=True, n_experts=8,
                   n_experts_per_tok=2, n_shared_experts=1),
    # MLA 版：验证低秩 KV 压缩对显存/带宽的收益
    # latent(48)+rope(32)=80 < GQA 的 n_kv_heads(2)×head_dim(48)=96 → KV 缓存更小
    "mla": _preset(dim=384, n_layers=12, n_heads=8, attn_type="mla",
                   mla_latent_dim=48, mla_rope_dim=32, vocab_size=4096),
    # 混合层：滑动窗口 + 线性注意力交替
    "hybrid": _preset(dim=384, n_layers=12, n_heads=8, n_kv_heads=2,
                      vocab_size=4096, attn_window=512,
                      layer_types="window,linear,window,full"),
    # 投机解码用的 draft 模型（层数少、头数少）
    "draft": _preset(dim=192, n_layers=4, n_heads=6, n_kv_heads=2, hidden_dim=512,
                     vocab_size=4096, attn_window=512),
    # 2026 混合架构：3 层 Gated DeltaNet + 1 层 Full Attention（Qwen3.6 式）
    "gdn_hybrid": _preset(dim=384, n_layers=12, n_heads=8, n_kv_heads=2,
                          vocab_size=4096, layer_types="gdn,gdn,gdn,full",
                          gdn_chunk_size=64),
    # 2026 混合架构：Mamba-2(SSD) 与 Full Attention 交替（Nemotron 3 Ultra 式）
    "mamba_hybrid": _preset(dim=384, n_layers=12, n_heads=8, n_kv_heads=2,
                            vocab_size=4096, layer_types="mamba2,mamba2,mamba2,full",
                            ssm_state_size=32),
}
