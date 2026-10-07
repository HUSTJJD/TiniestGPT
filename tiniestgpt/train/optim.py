"""优化器：AdamW / Muon / Sophia-G / Lion。

**Muon**（2024–2025 最值得关注的新优化器）：
它对**矩阵形状**的参数先用动量，再做一次 **Newton-Schulz 正交化**
（把更新方向变成近似正交矩阵），从而让每一步都在"所有方向上均匀前进"，
而不是被少数大奇异值方向主导。实践上：
  * 收敛所需 step 数明显少于 AdamW，尤其在小模型 / 小 batch 上；
  * 通常与 AdamW 混用：矩阵参数用 Muon，embedding / bias / norm 用 AdamW。

**Sophia-G**：用 Hutchinson 估计器估 Hessian 对角线，按曲率裁剪更新量，
对 LLM 的 loss 尖刺更鲁棒。

**Lion**：只用 sign 的动量法，内存只需动量一项（比 Adam 省一半）。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from ..common.registry import OPTIMIZER

__all__ = ["Muon", "SophiaG", "Lion", "build_optimizer", "newton_schulz_orthogonalize"]


# --------------------------------------------------------------------------- #
# Newton-Schulz 正交化（Muon 的核心）
# --------------------------------------------------------------------------- #
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 6, eps: float = 1e-7) -> torch.Tensor:
    """用五次迭代把 G 变成近似半正交矩阵（``X Xᵀ ≈ I``）。

    系数来自 Jordan 等人的实验：在 [0,1] 区间上拟合 ``x^(1/2)`` 的多项式，
    既不需要 SVD（太慢），也比一次幂迭代更精确。
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


# --------------------------------------------------------------------------- #
# Muon
# --------------------------------------------------------------------------- #
class Muon(torch.optim.Optimizer):
    """矩阵参数走 Muon，其余（1-D）参数走 AdamW。"""

    def __init__(self, params: Iterable, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 6, weight_decay: float = 0.0,
                 betas: tuple = (0.9, 0.95), eps: float = 1e-8, adamw_lr: Optional[float] = None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, betas=betas, eps=eps,
                        adamw_lr=adamw_lr if adamw_lr is not None else lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() >= 2:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    upd = g.add(buf, alpha=momentum) if group["nesterov"] else buf.clone()
                    upd = newton_schulz_orthogonalize(upd, steps=ns)
                    # 按 fan-out/fan-in 缩放，让不同形状矩阵的更新幅度可比
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(upd, alpha=-lr * scale)
                else:
                    # --- AdamW 兜底（embedding / bias / norm） ---
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bias1 = 1 - b1 ** state["step"]
                    bias2 = 1 - b2 ** state["step"]
                    step_size = group["adamw_lr"] * math.sqrt(bias2) / bias1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    if wd > 0:
                        p.mul_(1.0 - group["adamw_lr"] * wd)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


# --------------------------------------------------------------------------- #
# Sophia-G
# --------------------------------------------------------------------------- #
class SophiaG(torch.optim.Optimizer):
    """二阶裁剪优化器：用 Hutchinson 估计 Hessian 对角线。"""

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.965, 0.99),
                 rho: float = 0.03, weight_decay: float = 0.1, eps: float = 1e-12,
                 update_interval: int = 10):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        eps=eps, update_interval=update_interval)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _hessian_diag(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Hutchinson: E[u ⊙ (H u)] = diag(H)，用 Rademacher 向量采样一次。"""
        u = torch.randint_like(g, high=2, dtype=g.dtype) * 2 - 1
        hvg = torch.autograd.grad(
            (g * u).sum(), p, retain_graph=False, allow_unused=True)[0]
        if hvg is None:
            return torch.zeros_like(g)
        return u * hvg

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state.update(exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p),
                                 hessian=torch.zeros_like(p), step=0)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)

                # 每 update_interval 步更新一次曲率估计（省算力）
                if state["step"] % group["update_interval"] == 1 and p.requires_grad:
                    h = self._hessian_diag(p, g)
                    state["hessian"].mul_(group["rho"]).addcmul_(h, h, value=1 - group["rho"])

                denom = torch.clamp(state["hessian"], min=group["eps"])
                ratio = (exp_avg.abs() / denom).clamp(max=1.0)      # 曲率裁剪
                update = ratio * torch.sign(exp_avg)
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


# --------------------------------------------------------------------------- #
# Lion
# --------------------------------------------------------------------------- #
class Lion(torch.optim.Optimizer):
    """Lion：只存动量，更新量取 sign。"""

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = torch.sign(m.mul(b1).add(g, alpha=1 - b1))
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
OPTIMIZER.register("muon", Muon)
OPTIMIZER.register("adamw", torch.optim.AdamW)
OPTIMIZER.register("lion", Lion)
OPTIMIZER.register("sophia", SophiaG)


def _group_params(model: nn.Module, weight_decay: float, no_decay_1d: bool = True,
                  muon_lr: Optional[float] = None, lr: float = 1e-3) -> List[dict]:
    """把参数分成「矩阵参数」与「1-D 参数」两组（Muon/AdamW 混合使用的前提）。"""
    matrix, vector = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (matrix if p.ndim >= 2 else vector).append(p)
    groups: List[dict] = []
    if muon_lr is not None:
        groups.append({"params": matrix, "lr": muon_lr, "weight_decay": weight_decay})
        groups.append({"params": vector, "lr": lr, "weight_decay": 0.0 if no_decay_1d else weight_decay})
    else:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if no_decay_1d and p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(cfg, model: nn.Module):
    """按配置构建优化器（自动处理 Muon 的参数分组）。"""
    name = cfg.optimizer.lower()
    if name == "muon":
        groups = _group_params(model, cfg.weight_decay, cfg.no_decay_1d, cfg.muon_lr, cfg.lr)
        return Muon(groups, lr=cfg.muon_lr, momentum=cfg.momentum, ns_steps=cfg.ns_steps,
                    weight_decay=cfg.weight_decay, betas=cfg.betas,
                    adamw_lr=cfg.lr, eps=cfg.adamw_eps)
    if name == "sophia":
        groups = _group_params(model, cfg.weight_decay, cfg.no_decay_1d)
        return SophiaG(groups, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if name == "lion":
        groups = _group_params(model, cfg.weight_decay, cfg.no_decay_1d)
        return Lion(groups, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if name == "adamw":
        groups = _group_params(model, cfg.weight_decay, cfg.no_decay_1d)
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.adamw_eps,
                                 weight_decay=cfg.weight_decay, fused=fused)
    raise ValueError(f"未知优化器: {name}")
