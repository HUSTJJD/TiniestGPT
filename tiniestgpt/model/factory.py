"""模型工厂：名字 / 配置 / checkpoint → 模型实例。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from .config import PRESETS, ModelConfig
from .transformer import Transformer

__all__ = ["build_model", "load_model"]


def build_model(name_or_cfg: Union[str, ModelConfig] = "tiny", **overrides) -> Transformer:
    """按预设名或 ModelConfig 构建模型。"""
    if isinstance(name_or_cfg, str):
        if name_or_cfg not in PRESETS:
            raise KeyError(f"未知预设: {name_or_cfg}；可用: {sorted(PRESETS)}")
        cfg = ModelConfig(**{**vars(PRESETS[name_or_cfg]), **overrides})
    else:
        cfg = name_or_cfg
    return Transformer(cfg)


def load_model(path: str | Path, map_location: str = "cpu") -> Transformer:
    """从 checkpoint 加载（自动识别是否 DDP 包装过的前缀）。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt.get("model", ckpt)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    cfg = ckpt.get("config")
    if cfg is None:
        raise RuntimeError(f"{path} 中没有保存 ModelConfig")
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    model = Transformer(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_model] 缺失权重: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[load_model] 多余权重: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return model
