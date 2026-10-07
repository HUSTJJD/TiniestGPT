"""把 TiniestGPT checkpoint 导出成 **HF LLaMA 兼容格式**。

为什么要做这件事：
  * 我们的架构（RMSNorm + SwiGLU + GQA + RoPE）与 LLaMA 同构，
    因此只要把**参数名**和**权重布局**对齐，就能被 transformers / vLLM / llama.cpp
    等生态直接加载 —— 这是"自己训的模型 → 生产推理引擎"之间最实用的一座桥。

需要处理的三处差异：
  1. 命名：`tok_embeddings` → `model.embed_tokens`、`mixer.q_proj` → `self_attn.q_proj` …
  2. FFN：我们把 gate/up 融合成一个 `w_in`（一次 GEMM），HF 里是分开的
     `gate_proj` / `up_proj`，导出时按前半/后半切分；
  3. 配置：需要写成 HF `LlamaConfig` 的字段（含 YaRN 的 `rope_scaling`）。

不支持导出：MoE / MLA / 线性注意力层（LLaMA 没有对应结构）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from ..model.config import ModelConfig
from ..model.factory import load_model

__all__ = ["export_hf", "build_llama_config", "map_state_dict"]

_LLAMA_ARCH = "LlamaForCausalLM"


def _lossy_features(cfg: ModelConfig) -> list[str]:
    """列出"LLaMA 结构无法表达"的特性——导出的模型会与原模型数值不同。"""
    out = []
    if cfg.post_norm:
        out.append("post_norm（Sandwich norm）：LLaMA 没有残差后的归一化，其权重会被丢弃")
    if cfg.qk_norm:
        out.append("qk_norm：LLaMA 不对 Q/K 做归一化")
    if cfg.rope_partial < 1.0:
        out.append(f"rope_partial={cfg.rope_partial}：LLaMA 会对全部 head_dim 施加 RoPE")
    if cfg.attn_softcap > 0:
        out.append("attn_softcap：LLaMA 不做 logit soft-capping")
    types = set(cfg.layer_type_list())
    if types - {"full", "window"}:
        out.append(f"含非标准层类型 {sorted(types)}")
    if len(types) > 1:
        out.append(f"混合层类型 {sorted(types)}：LLaMA 的 sliding_window_pattern 只能表达固定周期")
    return out


def build_llama_config(cfg: ModelConfig) -> Dict:
    """把 ModelConfig 翻译成 HF LlamaConfig 的字段。"""
    if cfg.moe_enabled:
        raise ValueError("MoE 模型无法映射为 LLaMA 结构（请改用 arch 相同的 MoE 模型）")
    if cfg.attn_type == "mla":
        raise ValueError("MLA 无法映射为 LLaMA 结构")
    if "linear" in cfg.layer_type_list():
        raise ValueError("线性注意力层无法映射为 LLaMA 结构")

    rope_scaling = None
    if cfg.rope_type == "yarn" and cfg.rope_scaling > 1.0:
        rope_scaling = {
            "type": "yarn",
            "factor": float(cfg.rope_scaling),
            "original_max_position_embeddings": int(cfg.rope_original_max_len),
        }

    # 滑动窗口：LLaMA 支持 sliding_window + pattern（pattern=1 表示每层都用）
    types = set(cfg.layer_type_list())
    sliding_window = cfg.attn_window if (cfg.attn_window > 0 and types == {"window"}) else None

    return {
        "sliding_window": sliding_window,
        "sliding_window_pattern": 1 if sliding_window else None,
        "architectures": [_LLAMA_ARCH],
        "model_type": "llama",
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.hidden_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": rope_scaling,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,      # 我们总是显式导出 lm_head，避免歧义
        "torch_dtype": "float32",
        # --- 以下为本项目额外保留的元信息，HF 会忽略未知字段 ---
        "tiniestgpt_note": "exported from TiniestGPT (LLaMA-compatible subset)",
        "tiniestgpt_norm_type": cfg.norm_type,
        "tiniestgpt_attn_window": cfg.attn_window,
        "tiniestgpt_attn_sinks": cfg.attn_sinks,
        "tiniestgpt_post_norm": cfg.post_norm,
    }


def map_state_dict(sd: Dict[str, torch.Tensor], cfg: ModelConfig) -> Dict[str, torch.Tensor]:
    """把本项目的参数名映射为 HF LLaMA 的参数名。"""
    out: Dict[str, torch.Tensor] = {}
    seen_ptr: Dict[int, str] = {}

    def put(name: str, t: torch.Tensor) -> None:
        v = t.detach().to(torch.float32).contiguous()
        # tie_embeddings 时 lm_head 与 embedding 是同一块存储；
        # safetensors 拒绝保存共享张量，因此第二个出现者要 clone 一份。
        if v.data_ptr() in seen_ptr:
            v = v.clone()
        seen_ptr[v.data_ptr()] = name
        out[name] = v

    put("model.embed_tokens.weight", sd["tok_embeddings.weight"])

    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        put(f"model.layers.{i}.self_attn.q_proj.weight", sd[p + "mixer.q_proj.weight"])
        put(f"model.layers.{i}.self_attn.k_proj.weight", sd[p + "mixer.k_proj.weight"])
        put(f"model.layers.{i}.self_attn.v_proj.weight", sd[p + "mixer.v_proj.weight"])
        put(f"model.layers.{i}.self_attn.o_proj.weight", sd[p + "mixer.o_proj.weight"])

        put(f"model.layers.{i}.input_layernorm.weight", sd[p + "norm1.weight"])
        put(f"model.layers.{i}.post_attention_layernorm.weight", sd[p + "norm2.weight"])

        # FFN：w_in 是 [2*hidden, dim]，前半是 gate、后半是 up（见 mlp.FeedForward）
        w_in = sd[p + "ffn.w_in.weight"]
        hidden = w_in.shape[0] // 2
        put(f"model.layers.{i}.mlp.gate_proj.weight", w_in[:hidden])
        put(f"model.layers.{i}.mlp.up_proj.weight", w_in[hidden:])
        put(f"model.layers.{i}.mlp.down_proj.weight", sd[p + "ffn.w_out.weight"])

    put("model.norm.weight", sd["norm_f.weight"])
    # tie_embeddings 时 lm_head 与 embedding 共享同一块存储，这里显式复制一份
    put("lm_head.weight", sd.get("lm_head.weight", sd["tok_embeddings.weight"]))
    return out


def export_hf(checkpoint: str, out_dir: str, tokenizer_path: Optional[str] = None,
              dtype: str = "float32", strict: bool = False) -> Path:
    """导出 checkpoint 到 ``out_dir``，返回目录路径。

    :param strict: 若模型含 LLaMA 无法表达的特性（post-norm / qk-norm / 部分 RoPE …），
        直接报错而不是静默导出一个**数值不同**的模型。
    """
    model = load_model(checkpoint, map_location="cpu")
    cfg: ModelConfig = model.cfg
    sd = model.state_dict()

    lossy = _lossy_features(cfg)
    if lossy:
        msg = "以下特性无法映射到 LLaMA 结构，导出的模型与原模型**数值不同**：\n  - " \
              + "\n  - ".join(lossy)
        if strict:
            raise ValueError(msg)
        print("[export_hf] 警告: " + msg)
        print("[export_hf] 提示: 需要无损导出请用 post_norm=false qk_norm=false 训练，"
              "或加 --strict 强制报错。")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = build_llama_config(cfg)
    config["torch_dtype"] = dtype
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    weights = map_state_dict(sd, cfg)
    dt = getattr(torch, dtype, torch.float32)
    weights = {k: v.to(dt) for k, v in weights.items()}

    # 优先用 safetensors（HF/vLLM 的首选格式），缺失时退回 torch.save
    try:
        from safetensors.torch import save_file  # type: ignore

        save_file({k: v.contiguous() for k, v in weights.items()},
                  str(out / "model.safetensors"))
        weight_file = "model.safetensors"
    except ImportError:
        torch.save(weights, out / "pytorch_model.bin")
        weight_file = "pytorch_model.bin"

    # 分词器：HF 生态需要 tokenizer.json / vocab；我们把 BPE 词表落成 JSON 便于检查
    if tokenizer_path:
        import shutil

        dst = out / "tokenizer_src.json"
        if Path(tokenizer_path).exists():
            shutil.copy(tokenizer_path, dst)

    meta = {
        "source_checkpoint": str(checkpoint),
        "weight_file": weight_file,
        "num_params": sum(v.numel() for v in weights.values()),
        "architectures": [_LLAMA_ARCH],
    }
    (out / "export_info.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"[export_hf] {out}  ({meta['num_params'] / 1e6:.1f} M params, {weight_file})")
    return out
