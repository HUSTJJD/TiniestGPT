"""AWQ：Activation-aware Weight Quantization。

核心观察：**权重不是等重要的**。
那些对应"激活值大"的输入通道（salient channels）如果量化出错，损失远大于其它通道。
AWQ 不增加 bit 数，而是给每个输入通道学一个缩放 ``s``：

    y = W x = (W · diag(s)) · (diag(s)⁻¹ · x)

把 ``s`` 折进权重后再量化 → 重要通道的有效分辨率更高；
``s⁻¹`` 折进前一个归一化层的权重（因此**不引入任何额外计算**）。

与 GPTQ 的区别：GPTQ 用二阶误差补偿，AWQ 用等价变换；
AWQ 更快（无 Hessian）、对激活分布更鲁棒，
但 GPTQ 在极低 bit（2~3bit）下通常更准。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from .base import QuantizedLinear, pack_int4

__all__ = ["search_scales", "awq_quantize_model", "pseudo_quantize"]


def pseudo_quantize(W: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """模拟量化（round-to-nearest + 反量化），用于搜索阶段的损失评估。"""
    out_f, in_f = W.shape
    if group_size <= 0 or in_f % group_size != 0:
        group_size = in_f
    g = W.reshape(out_f, in_f // group_size, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(g / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(out_f, in_f)


@torch.no_grad()
def search_scales(W: torch.Tensor, X: torch.Tensor, bits: int = 4, group_size: int = 128,
                  n_grid: int = 20) -> torch.Tensor:
    """网格搜索每个输入通道的缩放系数。

    :param W: [out, in] 权重
    :param X: [N, in]   校准激活
    :return:  [in]      最优缩放
    """
    dev = W.device
    Wf = W.float()
    Xf = X.float()
    # 通道重要性：激活的二阶矩
    sx = Xf.pow(2).mean(0).sqrt().clamp(min=1e-8)          # [in]
    best_s = torch.ones_like(sx)
    best_loss = torch.full_like(sx, float("inf"))

    for alpha in torch.linspace(0.0, 1.0, n_grid, device=dev):
        s = sx.pow(alpha).clamp(min=1e-5)
        Wq = pseudo_quantize(Wf * s.unsqueeze(0), bits, group_size)
        loss = ((Xf / s.unsqueeze(0)) @ Wq.T - Xf @ Wf.T).pow(2).mean(0)   # [out]
        # 逐通道衡量：用该通道对整体损失的贡献近似
        contrib = loss.mean().expand_as(sx) if loss.numel() == 1 else \
            ((Xf / s.unsqueeze(0)).pow(2).mean(0) * (Wq - Wf).pow(2).mean(0))
        better = contrib < best_loss
        best_loss = torch.where(better, contrib, best_loss)
        best_s = torch.where(better, s, best_s)
    return best_s.to(W.dtype)


def _preceding_norm_name(module_name: str) -> str | None:
    """按命名约定推断"前一个归一化层"（用于折叠 s⁻¹）。"""
    if ".ffn." in module_name:
        return module_name.split(".ffn.")[0] + ".norm2"
    if ".mixer." in module_name:
        return module_name.split(".mixer.")[0] + ".norm1"
    return None


@torch.no_grad()
def awq_quantize_model(model: nn.Module, calib: Dict[str, torch.Tensor],
                       bits: int = 4, group_size: int = 128, n_grid: int = 20,
                       skip_names=("lm_head", "gate", "embeddings"), verbose: bool = True
                       ) -> nn.Module:
    """``calib``: {模块名: 拼接后的激活 [N, in]}。原地替换 nn.Linear。"""
    modules = dict(model.named_modules())
    n_done = 0
    for name, mod in list(modules.items()):
        if not isinstance(mod, nn.Linear) or any(s in name for s in skip_names):
            continue
        X = calib.get(name)
        if X is None or X.ndim != 2:
            continue
        s = search_scales(mod.weight.data, X, bits=bits, group_size=group_size, n_grid=n_grid)
        W_scaled = mod.weight.data.float() * s.unsqueeze(0)
        ql = QuantizedLinear.from_float(mod, bits=bits, group_size=group_size)
        # 用"缩放后的权重"重新量化，并把 s 折进权重
        from .base import quantize_per_channel

        q, sc = _group_quantize(W_scaled, group_size, bits)
        if bits == 4:
            ql.qweight = pack_int4(q).contiguous()
            ql.q_shape = q.shape
        else:
            ql.qweight = q.contiguous()
        ql.scales = sc.to(ql.scales.dtype)

        parent_name = name.rsplit(".", 1)[0]
        parent = modules.get(parent_name, model)
        setattr(parent, name.rsplit(".", 1)[1], ql)

        # 把 s⁻¹ 折进前一个归一化层的权重
        norm_name = _preceding_norm_name(name)
        norm = modules.get(norm_name) if norm_name else None
        if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
            norm.weight.data = norm.weight.data / s.to(norm.weight.dtype)
        n_done += 1
        if verbose:
            print(f"[awq] {name} -> int{bits} (scale folded into {norm_name})")
    if verbose:
        print(f"[awq] quantized {n_done} layers")
    return model


def _group_quantize(W: torch.Tensor, group_size: int, bits: int):
    out_f, in_f = W.shape
    if group_size <= 0 or in_f % group_size != 0:
        group_size = in_f
    g = W.reshape(out_f, in_f // group_size, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(g / scale).clamp(-qmax, qmax).to(torch.int8)
    return q.reshape(out_f, in_f), scale.squeeze(-1)
