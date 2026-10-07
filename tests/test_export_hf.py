"""HF 导出测试：命名映射、权重切分、配置字段、以及反向回灌的一致性。"""

import json

import torch

from tiniestgpt.inference.export_hf import build_llama_config, export_hf, map_state_dict
from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model


def _cfg(**kw):
    base = dict(vocab_size=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                hidden_dim=64, max_seq_len=128)
    base.update(kw)
    return ModelConfig(**base)


def test_llama_config_fields():
    cfg = _cfg()
    c = build_llama_config(cfg)
    assert c["architectures"] == ["LlamaForCausalLM"]
    assert c["hidden_size"] == 32
    assert c["num_attention_heads"] == 4
    assert c["num_key_value_heads"] == 2
    assert c["intermediate_size"] == 64
    assert c["rope_scaling"] is None          # 默认 rope_scaling=1 → 不启用 YaRN


def test_llama_config_yarn():
    cfg = _cfg(rope_type="yarn", rope_scaling=4.0, rope_original_max_len=128)
    c = build_llama_config(cfg)
    assert c["rope_scaling"]["type"] == "yarn"
    assert c["rope_scaling"]["factor"] == 4.0


def test_state_dict_mapping_and_split():
    m = build_model(_cfg())
    sd = m.state_dict()
    out = map_state_dict(sd, m.cfg)

    # 关键命名
    assert "model.embed_tokens.weight" in out
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert "model.layers.0.input_layernorm.weight" in out
    assert "model.layers.0.mlp.gate_proj.weight" in out
    assert "model.norm.weight" in out
    assert "lm_head.weight" in out

    # gate/up 必须是从融合的 w_in 正确切出来的
    w_in = sd["layers.0.ffn.w_in.weight"]
    hidden = w_in.shape[0] // 2
    assert torch.equal(out["model.layers.0.mlp.gate_proj.weight"], w_in[:hidden])
    assert torch.equal(out["model.layers.0.mlp.up_proj.weight"], w_in[hidden:])

    # 不应残留本项目内部命名
    assert not any(k.startswith("layers.") or k.startswith("tok_embeddings") for k in out)


def test_roundtrip_weights_back_to_model():
    """导出后再反灌回本项目模型，权重必须完全一致（证明映射没有丢/错位）。

    注意：必须用 post_norm=False 的模型，否则 post-norm 权重无法映射（有损导出）。
    """
    m = build_model(_cfg(post_norm=False))
    sd = m.state_dict()
    out = map_state_dict(sd, m.cfg)

    back = {}
    back["tok_embeddings.weight"] = out["model.embed_tokens.weight"]
    for i in range(m.cfg.n_layers):
        p = f"layers.{i}."
        for src, dst in (("self_attn.q_proj", "mixer.q_proj"),
                         ("self_attn.k_proj", "mixer.k_proj"),
                         ("self_attn.v_proj", "mixer.v_proj"),
                         ("self_attn.o_proj", "mixer.o_proj")):
            back[p + dst + ".weight"] = out[f"model.layers.{i}.{src}.weight"]
        back[p + "norm1.weight"] = out[f"model.layers.{i}.input_layernorm.weight"]
        back[p + "norm2.weight"] = out[f"model.layers.{i}.post_attention_layernorm.weight"]
        gate = out[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = out[f"model.layers.{i}.mlp.up_proj.weight"]
        back[p + "ffn.w_in.weight"] = torch.cat([gate, up], dim=0)
        back[p + "ffn.w_out.weight"] = out[f"model.layers.{i}.mlp.down_proj.weight"]
    back["norm_f.weight"] = out["model.norm.weight"]

    m2 = build_model(_cfg(post_norm=False))
    missing, unexpected = m2.load_state_dict(back, strict=False)
    # 只允许缺少 lm_head（tie 时它与 embedding 共享）
    assert [k for k in missing if k != "lm_head.weight"] == [], missing
    for k, v in back.items():
        assert torch.allclose(m2.state_dict()[k], v), k


def test_strict_mode_rejects_lossy_export(tmp_path):
    """post_norm/qk_norm 等 LLaMA 表达不了的特性，--strict 必须报错而不是静默丢失。"""
    import pytest

    m = build_model(_cfg(post_norm=True))
    ckpt = tmp_path / "lossy.pt"
    torch.save({"model": m.state_dict(), "config": vars(m.cfg)}, ckpt)
    with pytest.raises(ValueError, match="LLaMA"):
        export_hf(str(ckpt), str(tmp_path / "hf_lossy"), strict=True)
    # 非 strict 则允许导出（并打印警告）
    export_hf(str(ckpt), str(tmp_path / "hf_lossy2"), strict=False)


def test_export_writes_files(tmp_path):
    m = build_model(_cfg(post_norm=False))
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "config": vars(m.cfg)}, ckpt)

    out_dir = tmp_path / "hf"
    export_hf(str(ckpt), str(out_dir))

    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()
    info = json.loads((out_dir / "export_info.json").read_text(encoding="utf-8"))
    assert info["num_params"] > 0


def test_unsupported_architectures():
    import pytest

    with pytest.raises(ValueError):
        build_llama_config(_cfg(moe_enabled=True, n_experts=2, n_experts_per_tok=1))
    with pytest.raises(ValueError):
        build_llama_config(_cfg(attn_type="mla"))
