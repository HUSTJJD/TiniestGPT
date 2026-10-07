"""SmoothQuant：W8A8 量化的关键一步——**把难度从激活迁移到权重**。

问题：激活里有极少数"离群通道"（数值比均值大几十倍），
per-tensor 量化会被它们拉爆；但 per-token 量化又无法走 INT8 张量核。

解法：利用数学等价

    Y = (X · diag(s)⁻¹) · (diag(s) · W)

选取 ``s_j = max|X_j|^α / max|W_j|^(1-α)``（α≈0.5）后，
激活的离群值被压平、权重稍微变难——而**权重本来就比激活好量化**。
推理时 ``s⁻¹`` 折进前一个 LayerNorm 的权重，同样零额外开销。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import QuantizedLinear, quantize_per_channel

__all__ = ["compute_smooth_scales", "smoothquant_quantize_model", "SmoothQuantLinear"]


@torch.no_grad()
def compute_smooth_scales(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """``s = max|X|^α / max|W|^(1-α)``，按输入通道计算。"""
    act = X.detach().float().abs().amax(dim=0).clamp(min=1e-8)     # [in]
    wgt = W.detach().float().abs().amax(dim=0).clamp(min=1e-8)     # [in]
    return (act.pow(alpha) / wgt.pow(1 - alpha)).clamp(min=1e-5)


class SmoothQuantLinear(nn.Module):
    """W8A8：权重 per-channel int8，激活 per-token int8，累加用 int32。"""

    def __init__(self, linear: nn.Linear, alpha: float = 0.5,
                 calib: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        W = linear.weight.data.float()
        if calib is not None:
            s = compute_smooth_scales(calib, W, alpha)
            W = W * s.unsqueeze(0)
        else:
            s = torch.ones(W.shape[1], device=W.device)
        self.register_buffer("smooth_scale", s.to(linear.weight.dtype))
        q, sc = quantize_per_channel(W, bits=8)
        self.register_buffer("qweight", q.contiguous())
        self.register_buffer("w_scale", sc.to(linear.weight.dtype))
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .base import quantize_per_token

        # 激活：per-token 量化（抵抗剩下的一点离群）
        xq, xs = quantize_per_token(x.float())
        w = self.qweight.to(xs.dtype) * self.w_scale
        out = torch.nn.functional.linear(xq.to(xs.dtype), w, None) * xs
        # 把 smooth_scale 折回去（等价于在前一层做过除法）
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


def _preceding_norm_name(module_name: str) -> str | None:
    if ".ffn." in module_name:
        return module_name.split(".ffn.")[0] + ".norm2"
    if ".mixer." in module_name:
        return module_name.split(".mixer.")[0] + ".norm1"
    return None


@torch.no_grad()
def smoothquant_quantize_model(model: nn.Module, calib: Dict[str, torch.Tensor],
                               alpha: float = 0.5,
                               skip_names=("lm_head", "gate", "embeddings"),
                               verbose: bool = True) -> nn.Module:
    """``calib``: {模块名: 激活 [N, in]}；把 s⁻¹ 折进前一个归一化层。"""
    modules = dict(model.named_modules())
    n = 0
    for name, mod in list(modules.items()):
        if not isinstance(mod, nn.Linear) or any(s in name for s in skip_names):
            continue
        X = calib.get(name)
        parent_name = name.rsplit(".", 1)[0]
        parent = modules.get(parent_name, model)
        if X is None:
            from .base import QuantizedLinear

            setattr(parent, name.rsplit(".", 1)[1],
                    QuantizedLinear.from_float(mod, bits=8))
            n += 1
            continue

        s = compute_smooth_scales(X, mod.weight.data, alpha)
        mod.weight.data = mod.weight.data.float() * s.unsqueeze(0)

        norm_name = _preceding_norm_name(name)
        norm = modules.get(norm_name) if norm_name else None
        if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
            norm.weight.data = norm.weight.data / s.to(norm.weight.dtype)

        setattr(parent, name.rsplit(".", 1)[1], SmoothQuantLinear(mod, alpha=alpha))
        n += 1
        if verbose:
            print(f"[smoothquant] {name} -> W8A8 (alpha={alpha})")
    if verbose:
        print(f"[smoothquant] quantized {n} layers")
    return model
