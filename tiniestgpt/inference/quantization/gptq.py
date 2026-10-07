"""GPTQ：一次性（one-shot）后训练量化。

思想来自 **OBD / OBS**（Optimal Brain Surgeon）：
量化第 i 列权重会带来误差 ``δ``，我们**不等着不管**，
而是把误差按二阶信息（Hessian 逆）**分摊到后面尚未量化的列**上，
让整体重建误差最小：

    δ = (w_i - q_i) / [H⁻¹]_ii
    W[:, i:] -= δ · H⁻¹[i, i:]

Hessian 用校准数据估计：``H = 2 · Σ x xᵀ``（对 MSE 损失而言与输入无关）。
由于我们只需要 H⁻¹，用 Cholesky 求逆既稳又快；
``percdamp`` 给对角线加阻尼防止奇异。

实践参数：4bit + group_size=128 + 128~256 条校准样本，
在 7B 模型上精度损失通常 < 1%。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .base import QuantizedLinear, pack_int4, unpack_int4

__all__ = ["gptq_quantize_linear", "gptq_quantize_model"]


@torch.no_grad()
def _hessian(calib: Sequence[torch.Tensor], n_cols: int, device) -> torch.Tensor:
    H = torch.zeros(n_cols, n_cols, device=device, dtype=torch.float32)
    n = 0
    for x in calib:
        xf = x.detach().float().reshape(-1, n_cols).to(device)
        if xf.shape[0] == 0:
            continue
        H += 2.0 * (xf.T @ xf)
        n += xf.shape[0]
    if n == 0:
        return torch.eye(n_cols, device=device)
    return H / n


@torch.no_grad()
def gptq_quantize_linear(linear: nn.Linear, calib: Sequence[torch.Tensor],
                         bits: int = 4, group_size: int = 128,
                         percdamp: float = 0.01, blocksize: int = 128,
                         actorder: bool = True) -> QuantizedLinear:
    dev = linear.weight.device
    W = linear.weight.data.detach().float().to(dev)       # [out, in]
    out_f, in_f = W.shape
    if group_size <= 0:
        group_size = in_f

    H = _hessian(calib, in_f, dev)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # 重要性排序（act-order）：先量化 Hessian 对角线大的列，误差更小
    if actorder:
        order = torch.argsort(torch.diag(H), descending=True)
        W = W[:, order]
        H = H[order][:, order]
        inv_order = torch.argsort(order)
    else:
        order = None

    damp = percdamp * torch.mean(torch.diag(H))
    H[torch.arange(in_f), torch.arange(in_f)] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)

    # 每组的量化 scale（按原始权重的最大绝对值）
    scales = W.abs().amax(dim=0)                                  # [in]
    gs = group_size
    scales = scales.reshape(in_f // gs, gs).amax(dim=1, keepdim=True) \
        .clamp(min=1e-12) / (2 ** (bits - 1) - 1)                 # [n_groups, 1]
    scale_per_col = scales.expand(in_f // gs, gs).reshape(in_f)    # [in]

    Q = torch.zeros_like(W)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(cnt):
            w = W1[:, i]
            d = Hinv1[i, i]
            sc = scale_per_col[i1 + i]
            q = torch.round(w / sc).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            Q1[:, i] = q
            err = (w - q * sc) / d                                 # 反量化后的重建误差
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if order is not None:
        Q = Q[:, inv_order]
        scale_per_col = scale_per_col[inv_order]

    # 组装 QuantizedLinear
    ql = QuantizedLinear.from_float(linear, bits=bits, group_size=group_size)
    q_int = Q.round().to(torch.int8)
    if bits == 4:
        ql.qweight = pack_int4(q_int).contiguous()
        ql.q_shape = q_int.shape
    else:
        ql.qweight = q_int.contiguous()
    ql.scales = scale_per_col.reshape(in_f // gs, gs).amax(dim=1).to(ql.scales.dtype)
    return ql


@torch.no_grad()
def gptq_quantize_model(model: nn.Module, calib: dict, bits: int = 4, group_size: int = 128,
                        skip_names=("lm_head", "gate", "embeddings"), verbose: bool = True
                        ) -> nn.Module:
    """按模块名批量做 GPTQ。``calib``: {模块名: [激活张量, ...]}"""
    from typing import Dict

    def _recurse(module: nn.Module, prefix: str = "") -> int:
        n = 0
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(s in name for s in skip_names) or full not in calib:
                    continue
                setattr(module, name, gptq_quantize_linear(child, calib[full], bits, group_size))
                n += 1
                if verbose:
                    print(f"[gptq] {full} -> int{bits}")
            else:
                n += _recurse(child, full)
        return n

    total = _recurse(model)
    if verbose:
        print(f"[gptq] quantized {total} layers")
    return model
