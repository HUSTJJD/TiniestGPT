"""MuonClip / QK-Clip：直接约束 attention logit，而不是约束梯度。

Kimi K2 在 15.5T token 的预训练中做到**零 loss spike**，靠的就是这个技巧。
它的洞察是：Muon 的更新是"正交化"的，会让权重范数稳定增长，
于是某些 attention head 的 logit 越来越大::

    S_max_h = max_t(Q_h K_hᵀ / sqrt(d))

一旦 ``S_max`` 远超阈值，softmax 会极度饱和（one-hot 化），
梯度消失 + 数值放大 → loss spike。

**梯度裁剪在这里没用**——它限制的是"更新量"，而 QK-Clip 限制的是
"造成饱和的前向 logit 本身"。做法是按 head 计算::

    gamma_h = min(1, tau / S_max_h)
    W_Q[h] <- gamma_h^alpha       · W_Q[h]
    W_K[h] <- gamma_h^(1 - alpha) · W_K[h]

alpha=0.5 时 Q/K 各承担平方根比例的缩放，二者点积整体约乘 ``gamma``，
**不改变该 head 的相对分布，只把绝对值拉回安全区**。

使用方式::

    guard = QKClipGuard(model, tau=100.0)
    guard.install()          # 注册 hook，前向时顺带观测 S_max（零额外前向代价）
    ...
    guard.observe()          # 每个 forward 之后调用
    stats = guard.clip()     # 每 N 步调用一次
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

__all__ = ["QKClipGuard", "max_attn_logits", "apply_qk_clip"]


def _as_heads(q: torch.Tensor, k: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 q/k 投影输出整理成 ``[..., H, D]``。

    GQA 下 q 是 ``n_heads·D`` 而 k 只有 ``n_kv_heads·D``，
    必须先把 K 头广播（repeat_kv）到与 Q 对齐，否则逐 head 的 logit 是错的。
    """
    D = q.shape[-1] // H
    qh = q.reshape(*q.shape[:-1], H, D)
    hk = k.shape[-1] // D
    kh = k.reshape(*k.shape[:-1], hk, D)
    if hk < H:
        kh = kh.repeat_interleave(H // hk, dim=-2)
    elif hk > H:
        kh = kh[..., :H, :]
    return qh, kh


def _find_qk_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """找出所有带 ``q_proj`` / ``k_proj`` 的模块（Attention / MLA / GDN / Retention）。"""
    out: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        q, k = getattr(m, "q_proj", None), getattr(m, "k_proj", None)
        if isinstance(q, nn.Linear) and isinstance(k, nn.Linear):
            out[name] = m
    return out


def max_attn_logits(model: nn.Module, input_ids: torch.Tensor,
                    head_dim: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """跑一次前向，用 hook 抓出每个模块**逐 head** 的最大 attention logit。

    返回 ``{模块名: [H]}``。RoPE 是保长旋转，不改端点积的上界量级，
    因此这里直接用投影后的 q/k 估计（省一次 RoPE，误差在可接受范围）。
    """
    mods = _find_qk_modules(model)
    buffers: Dict[str, Dict[str, torch.Tensor]] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make(name: str, key: str):
        def hook(_mod, _inp, out):
            buffers.setdefault(name, {})[key] = out.detach()
        return hook

    for name, m in mods.items():
        handles.append(m.q_proj.register_forward_hook(make(name, "q")))
        handles.append(m.k_proj.register_forward_hook(make(name, "k")))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()

    out: Dict[str, torch.Tensor] = {}
    for name, buf in buffers.items():
        q, k = buf.get("q"), buf.get("k")
        if q is None or k is None:
            continue
        m = mods[name]
        H = getattr(m, "n_heads", None) or getattr(m, "num_heads", None)
        if not H or q.shape[-1] % H:
            continue
        qh, kh = _as_heads(q, k, H)
        D = qh.shape[-1]
        scores = torch.einsum("...thd,...shd->...hts", qh, kh) / (D ** 0.5)
        out[name] = scores.amax(dim=(-1, -2)).mean(dim=0)      # [H]：对 batch 取均值
    return out


def apply_qk_clip(model: nn.Module, s_max: Dict[str, torch.Tensor],
                  tau: float, alpha: float = 0.5) -> Dict[str, float]:
    """按 head 缩放 ``W_Q`` / ``W_K``，返回每个模块的实际缩放统计。"""
    mods = _find_qk_modules(model)
    stats: Dict[str, float] = {}
    for name, s in s_max.items():
        m = mods.get(name)
        if m is None:
            continue
        H = getattr(m, "n_heads", None) or getattr(m, "num_heads", None)
        D = m.q_proj.weight.shape[0] // H
        gamma = torch.clamp(tau / s.clamp(min=1e-6), max=1.0).to(m.q_proj.weight.device)  # [H]
        sq = (gamma ** alpha).repeat_interleave(D)            # [H*D]

        # GQA 下 K 的头数少于 Q：一个 K 头服务 group 个 Q 头，
        # 取组内**最严格**（最小）的 gamma，保证它服务的每个 Q 头都被约束到。
        n_kv = m.k_proj.weight.shape[0] // D
        group = max(H // max(n_kv, 1), 1)
        gk = gamma.view(n_kv, group).amin(dim=1) if n_kv * group == H else gamma[:n_kv]
        sk = (gk ** (1.0 - alpha)).repeat_interleave(D)

        with torch.no_grad():
            m.q_proj.weight.mul_(sq.unsqueeze(1))
            m.k_proj.weight.mul_(sk.unsqueeze(1))
            if m.q_proj.bias is not None:
                m.q_proj.bias.mul_(sq)
            if m.k_proj.bias is not None:
                m.k_proj.bias.mul_(sk)
        stats[name] = float(gamma.min().item())
    return stats


class QKClipGuard:
    """训练期常驻的 QK-Clip 守卫。

    :param tau: logit 阈值；超过就按比例把 Q/K 拉回来
    :param alpha: Q/K 的缩放分配（0.5 = 各担一半）
    :param every: 每多少步执行一次 clip
    """

    def __init__(self, model: nn.Module, tau: float = 100.0, alpha: float = 0.5,
                 every: int = 50, head_dim: Optional[int] = None) -> None:
        self.model = model
        self.tau = tau
        self.alpha = alpha
        self.every = every
        self.mods = _find_qk_modules(model)
        self.running: Dict[str, torch.Tensor] = {}
        self.history: List[Dict[str, float]] = []
        self._enabled = True

    # ------------------------------------------------------------------ #
    def install(self) -> "QKClipGuard":
        """注册 hook：前向时顺手记录 q/k（不额外跑前向）。

        注意 q/k 两个 hook 必须**共用一个 buffer**，否则各自只看到一半，
        "两个都到齐了"这个条件永远不成立。
        """
        self._buffers = {}
        for name, m in self.mods.items():
            buf: Dict[str, torch.Tensor] = {}
            self._buffers[name] = buf
            m.q_proj.register_forward_hook(self._hook(name, "q", buf))
            m.k_proj.register_forward_hook(self._hook(name, "k", buf))
        return self

    def _hook(self, name: str, key: str, buf: Dict[str, torch.Tensor]):
        def fn(_mod, _inp, out):
            if not self._enabled:
                return
            buf[key] = out.detach()
            if "q" in buf and "k" in buf:
                self._record(name, buf["q"], buf["k"])
                buf.clear()
        return fn

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _record(self, name: str, q: torch.Tensor, k: torch.Tensor) -> None:
        m = self.mods[name]
        H = getattr(m, "n_heads", None) or getattr(m, "num_heads", None)
        if not H or q.shape[-1] % H:
            return
        qh, kh = _as_heads(q, k, H)
        D = qh.shape[-1]
        scores = torch.einsum("...thd,...shd->...hts", qh, kh) / (D ** 0.5)
        s = scores.amax(dim=(-1, -2)).amax(dim=0)          # [H]，取 batch 内最大
        prev = self.running.get(name)
        self.running[name] = s if prev is None else torch.maximum(prev, s.to(prev.device))

    def observe(self) -> None:
        """显式观测入口（hook 模式下为空操作，保留给"探针式"用法）。"""
        return None

    # ------------------------------------------------------------------ #
    def clip(self, force: bool = False) -> Optional[Dict[str, float]]:
        """执行一次 clip，并清空运行期最大值。"""
        if not self.running:
            return None
        stats = apply_qk_clip(self.model, self.running, self.tau, self.alpha)
        worst = max(1.0 / max(v, 1e-6) for v in stats.values()) if stats else 1.0
        self.running.clear()
        rec = {"n_modules": len(stats), "min_gamma": min(stats.values()) if stats else 1.0,
               "max_shrink": worst}
        self.history.append(rec)
        if rec["min_gamma"] < 0.999:
            log.info("[qk-clip] 触发: %d 个模块, 最小 gamma=%.3f", rec["n_modules"], rec["min_gamma"])
        return rec

    def maybe_clip(self, step: int) -> Optional[Dict[str, float]]:
        if step > 0 and step % self.every == 0:
            return self.clip()
        return None

    def report(self) -> str:
        if not self.history:
            return "qk-clip: 未触发过（S_max 始终低于 tau=%.0f）" % self.tau
        n = sum(1 for h in self.history if h["min_gamma"] < 0.999)
        return (f"qk-clip: 检查 {len(self.history)} 次, 触发 {n} 次, "
                f"最小 gamma={min(h['min_gamma'] for h in self.history):.3f}")
