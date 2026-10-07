"""量化基础：把浮点矩阵变成「整数 + 缩放因子」。

三条主线：
  * **W8A8**（SmoothQuant）：权重与激活都量化，走 INT8 张量核，延迟收益最大；
  * **W4A16**（GPTQ / AWQ）：只量化权重到 4bit，激活保持 fp16。
    这是 **memory-bound 的 decode 阶段** 的最优解——
    因为 decode 的瓶颈是"把权重从显存搬到计算单元"，
    权重减半直接让带宽需求减半；
  * **NF4**（QLoRA）：信息论最优的 4bit 浮点码本 + 双重量化，
    在不微调的情况下保住精度。

量化粒度（越细越准、开销越大）：per-tensor < per-channel < per-group < per-token。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantScheme", "quantize_absmax", "dequantize", "quantize_per_channel",
    "quantize_per_token", "pack_int4", "unpack_int4", "nf4_quantize", "nf4_dequantize",
    "QuantizedLinear", "quantize_model", "fp8_quantize", "fp8_dequantize",
]


@dataclass
class QuantScheme:
    bits: int = 8                    # 4 | 8
    group_size: int = -1             # -1 = per-channel；>0 = per-group
    symmetric: bool = True           # 对称（零点固定 0）
    granularity: str = "channel"     # tensor | channel | group | token
    dtype: str = "int"               # int | nf4 | fp8


# --------------------------------------------------------------------------- #
# 基础量化
# --------------------------------------------------------------------------- #
def quantize_absmax(x: torch.Tensor, bits: int = 8, dim: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 absmax 量化：``q = round(x / scale)``，``scale = max|x| / qmax``。"""
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        scale = x.abs().max() / qmax
    else:
        scale = x.abs().amax(dim=dim, keepdim=True) / qmax
    scale = scale.clamp(min=1e-12)
    q = torch.round(x / scale).clamp(-qmax - (0 if bits > 4 else 0), qmax)
    return q.to(torch.int8 if bits == 8 else torch.int8), scale.to(x.dtype)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale


def quantize_per_channel(w: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """按输出通道量化（out_features 维）——LLM 权重量化的默认粒度。"""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.to(w.dtype)


def quantize_per_token(x: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 token 量化激活（W8A8 中的 A8）：每个 token 一个 scale，抵抗激活离群值。"""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.to(x.dtype)


# --------------------------------------------------------------------------- #
# INT4 打包
# --------------------------------------------------------------------------- #
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """把 int8（取值 -8..7）压成 int32（每字节存 2 个 nibble）。最后一维长度需为偶数。"""
    qi = (q.to(torch.int32) + 8) & 0xF
    lo = qi[..., 0::2]
    hi = qi[..., 1::2]
    return (lo | (hi << 4)).to(torch.int32).contiguous()


def unpack_int4(packed: torch.Tensor, out_shape: Tuple[int, ...]) -> torch.Tensor:
    lo = (packed & 0xF).to(torch.int8)
    hi = ((packed >> 4) & 0xF).to(torch.int8)
    out = torch.stack([lo - 8, hi - 8], dim=-1)
    return out.reshape(out_shape)


# --------------------------------------------------------------------------- #
# NF4（QLoRA）
# --------------------------------------------------------------------------- #
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(w: torch.Tensor, block_size: int = 64, double_quant: bool = True):
    """NF4：先按 block 做 absmax 归一化，再映射到信息论最优的 4bit 码本。"""
    dev, dt = w.device, w.dtype
    levels = NF4_LEVELS.to(dev)
    wf = w.float()
    orig_shape = wf.shape
    flat = wf.reshape(-1)
    n = flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    normed = blocks / absmax

    # 找最近的码本下标（16 个级别，直接广播即可）
    idx = torch.argmin((normed.unsqueeze(-1) - levels) ** 2, dim=-1).to(torch.uint8)

    # 双重量化：absmax 本身再量化到 8bit
    if double_quant:
        a_scale = absmax.max() / 127.0
        a_q = torch.round(absmax / a_scale.clamp(min=1e-12)).to(torch.uint8)
        absmax_deq = a_q.float() * a_scale
    else:
        a_q = None
        a_scale = None
        absmax_deq = absmax
    return {"codes": idx.reshape(-1)[:n], "absmax_q": a_q, "absmax_scale": a_scale,
            "block_size": block_size, "orig_shape": orig_shape, "dtype": dt}


def nf4_dequantize(q: dict) -> torch.Tensor:
    levels = NF4_LEVELS.to(q["codes"].device)
    codes = q["codes"].long()
    bs = q["block_size"]
    n = codes.numel()
    pad = (bs - n % bs) % bs
    flat = torch.cat([codes, codes.new_zeros(pad)]) if pad else codes
    blocks = flat.reshape(-1, bs)
    if q["absmax_q"] is not None:
        absmax = q["absmax_q"].float() * float(q["absmax_scale"])
    else:
        absmax = None
    out = levels[blocks]
    if absmax is not None:
        out = out * absmax
    return out.reshape(-1)[:n].reshape(q["orig_shape"]).to(q["dtype"])


# --------------------------------------------------------------------------- #
# FP8 (E4M3)
# --------------------------------------------------------------------------- #
def fp8_quantize(x: torch.Tensor, per_tensor: bool = True):
    """E4M3 量化：4 位指数 + 3 位尾数，动态范围足够、硬件支持好（H100 / 4090）。"""
    f8 = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None
    if f8 is None:
        return x.half(), torch.tensor(1.0, device=x.device)
    finfo = torch.finfo(f8)
    if per_tensor:
        scale = x.abs().amax().clamp(min=1e-12) / finfo.max
        return (x / scale).clamp(finfo.min, finfo.max).to(f8), scale.to(x.dtype)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / finfo.max
    return (x / scale).clamp(finfo.min, finfo.max).to(f8), scale.to(x.dtype)


def fp8_dequantize(x, scale: torch.Tensor) -> torch.Tensor:
    return x.to(scale.dtype) * scale


# --------------------------------------------------------------------------- #
# 量化线性层
# --------------------------------------------------------------------------- #
class QuantizedLinear(nn.Module):
    """把 nn.Linear 换成「整数权重 + scale」的形式，forward 时反量化后做 GEMM。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 bits: int = 8, group_size: int = -1, symmetric: bool = True,
                 use_fp8: bool = False, device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.use_fp8 = use_fp8

        self.register_buffer("qweight", torch.zeros(1, dtype=torch.int8, device=device))
        self.register_buffer("scales", torch.zeros(1, dtype=dtype, device=device))
        self.register_buffer("zeros", torch.zeros(1, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 8, group_size: int = -1,
                   symmetric: bool = True, use_fp8: bool = False,
                   weight: Optional[torch.Tensor] = None) -> "QuantizedLinear":
        W = (weight if weight is not None else linear.weight.data).float()
        out_f, in_f = W.shape
        m = cls(in_f, out_f, bias=linear.bias is not None, bits=bits,
                group_size=group_size, symmetric=symmetric, use_fp8=use_fp8,
                device=W.device, dtype=W.dtype)
        if use_fp8:
            q, s = fp8_quantize(W, per_tensor=(group_size <= 0))
            m.qweight = q.view(torch.int8).contiguous()
            m.scales = s.detach().to(W.dtype)
            m.is_fp8 = True
        elif bits == 8:
            q, s = quantize_per_channel(W, bits=8) if group_size <= 0 else cls._group_quant(W, group_size, 8)
            m.qweight = q.contiguous()
            m.scales = s.detach().to(W.dtype)
            m.zeros = torch.zeros_like(s)
        elif bits == 4:
            q, s = cls._group_quant(W, group_size if group_size > 0 else in_f, 4)
            m.qweight = pack_int4(q).contiguous()
            m.scales = s.detach().to(W.dtype)
            m.q_shape = q.shape
        else:
            raise ValueError("仅支持 4 / 8 bit")
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.detach().to(W.dtype))
        return m

    @staticmethod
    def _group_quant(W: torch.Tensor, group_size: int, bits: int):
        """按 group_size 分组量化（GPTQ/AWQ 常用 128）。"""
        out_f, in_f = W.shape
        if in_f % group_size != 0:
            group_size = in_f
        g = W.reshape(out_f, in_f // group_size, group_size)
        qmax = 2 ** (bits - 1) - 1
        scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
        q = torch.round(g / scale).clamp(-qmax, qmax).to(torch.int8)
        return q.reshape(out_f, in_f), scale.squeeze(-1)

    # ------------------------------------------------------------------ #
    def dequantize_weight(self) -> torch.Tensor:
        if getattr(self, "is_fp8", False):
            return fp8_dequantize(self.qweight.view(torch.float8_e4m3fn), self.scales)
        if self.bits == 4:
            q = unpack_int4(self.qweight, self.q_shape).to(self.scales.dtype)
        else:
            q = self.qweight.to(self.scales.dtype)
        if self.group_size > 0 or self.qweight.numel() != self.qweight.shape[0] * self.qweight.shape[1]:
            # group 量化：scales 形状 [out, n_groups]
            g = self.group_size if self.group_size > 0 else self.in_features
            w = q.reshape(self.out_features, self.in_features // g, g) * self.scales.unsqueeze(-1)
            return w.reshape(self.out_features, self.in_features)
        return q * self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # INT4 且打开了融合内核时走"寄存器里反量化"的路径：
        # 权重不再 materialize 成 fp16，省掉一次写 + 一次读（见 kernels/int4_gemm.py）。
        if self.bits == 4 and self._fuse_enabled():
            from ..kernels.int4_gemm import fused_int4_gemm

            return fused_int4_gemm(self.qweight, self.scales, x, self.group_size,
                                   self.q_shape, self.bias)
        w = self.dequantize_weight().to(x.dtype)
        return F.linear(x, w, self.bias)

    def _fuse_enabled(self) -> bool:
        flag = getattr(self, "use_fused_kernel", None)
        if flag is None:
            try:
                from ..kernels.int4_gemm import fused_int4_enabled

                return fused_int4_enabled()
            except Exception:  # pragma: no cover
                return False
        return bool(flag)

    def extra_repr(self) -> str:
        return f"({self.out_features}, {self.in_features}), bits={self.bits}, group={self.group_size}"


# --------------------------------------------------------------------------- #
def quantize_model(model: nn.Module, bits: int = 8, group_size: int = -1,
                   skip_names=("lm_head", "gate", "embeddings"),
                   use_fp8: bool = False, verbose: bool = False) -> nn.Module:
    """递归把 nn.Linear 换成 QuantizedLinear（原地修改）。

    ``skip_names`` 默认跳过 lm_head / MoE gate / embedding——
    经验上这些层量化后精度损失最明显。
    """
    n = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip_names):
                continue
            setattr(model, name, QuantizedLinear.from_float(
                module, bits=bits, group_size=group_size, use_fp8=use_fp8))
            n += 1
        else:
            quantize_model(module, bits, group_size, skip_names, use_fp8)
    if verbose and n:
        print(f"[quantize] replaced {n} Linear modules at level {type(model).__name__}")
    return model
