"""2026 年的两种"线性序列混合器"：**Gated DeltaNet** 与 **Mamba-2 / SSD**。

它们要解决的都是同一个问题：Full Attention 的 KV Cache 随上下文线性增长，
decode 每步都要把历史 K/V 从 HBM 读一遍。线性混合器把历史压进一个**固定大小的状态**，
于是 decode 的成本与上下文长度无关。

但两者"压缩"的方式完全不同，这是 2026 年最容易被混淆的一组概念：

* **Gated DeltaNet（Qwen3.6 / Kimi Linear）**
  状态是一个矩阵 ``S ∈ R^{Dv×Dk}``，用 **Delta Rule** 更新::

      prediction = S_{t-1} k_t
      error      = v_t - prediction
      S_t = α_t · S_{t-1} + β_t · error ⊗ k_t
      o_t = S_t q_t

  与朴素线性注意力（RetNet，只做 ``S += k vᵀ`` 累加）的区别在于
  **同一个 key 再次出现时能"改写"旧关联而不是继续累加**——这是它召回精度高的原因。
  ``α_t`` 是遗忘门，``β_t`` 是"在线学习率"。

* **Mamba-2 / SSD（Nemotron 3 Ultra / Falcon-H1R）**
  状态同样是固定大小，但衰减是**逐通道的标量**，更新是纯外积累加::

      h_t = a_t · h_{t-1} + b_t ⊗ x_t
      y_t = Σ_n c_t[n] · h_t[n, :]

  没有 delta rule 的"改写"能力，但结构更简单——**块内可以完全用矩阵乘并行**
  （GDN 的块内并行需要 UT/WY 变换与三角求解，本项目标记为后续工作）。

复杂度对照（L=上下文，C=块大小，D=head_dim，N=状态维度）：

========================  ==================  ==================  ================
结构                      prefill             decode 每步          随 L 增长的状态
========================  ==================  ==================  ================
Full Attention            O(L²)               O(L)                O(L) 的 KV
Sliding Window            O(L·W)              O(W)                O(W) 的 KV
Gated DeltaNet            O(L·C·D)            O(D²)               **固定矩阵 S**
Mamba-2 / SSD             O(L·C·N)            O(N·D)              **固定状态 h**
========================  ==================  ==================  ================

三种形态的设计原则（与本项目其余模块一致）：
``naive``（逐步循环，可读性优先）≡ ``chunk``（分块，数值稳定）≡ ``recurrent``（decode）。
三者由 ``tests/test_sequence_mixer.py`` 做数值一致性校验。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norms import RMSNorm

__all__ = ["GatedDeltaNet", "Mamba2Mixer", "build_sequence_mixer"]


# =========================================================================== #
#  Gated DeltaNet
# =========================================================================== #
class GatedDeltaNet(nn.Module):
    """Gated DeltaNet：Delta Rule + 遗忘门的线性注意力。

    三种形态：

    * :meth:`forward_sequential` —— 逐步递推（参考实现，最直白，O(L) 串行步）；
    * :meth:`forward_chunk`      —— 分块扫描（每个块内重新开始累积衰减连乘
      → **长序列不会下溢**，这也是真实 kernel 必须分块的根本原因）；
    * :meth:`forward_recurrent`  —— 单步 decode（生成时使用，O(D²) 且不需要 KV Cache）。
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.chunk_size = cfg.gdn_chunk_size
        H, D = cfg.n_heads, cfg.head_dim
        C = cfg.dim

        self.q_proj = nn.Linear(C, H * D, bias=False)
        self.k_proj = nn.Linear(C, H * D, bias=False)
        self.v_proj = nn.Linear(C, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, C, bias=False)
        # β = 在线学习率（sigmoid → (0,1)）；α = 遗忘门（sigmoid → (0,1)）
        self.b_proj = nn.Linear(C, H, bias=True)
        self.a_proj = nn.Linear(C, H, bias=True)
        nn.init.zeros_(self.b_proj.weight)
        nn.init.constant_(self.b_proj.bias, -2.0)      # β≈0.12，初始接近"少改写"
        nn.init.zeros_(self.a_proj.weight)
        nn.init.constant_(self.a_proj.bias, 2.9)       # α≈0.95，初始接近"少遗忘"
        self.norm = RMSNorm(D)

        if cfg.gdn_conv_kernel > 0:
            k = cfg.gdn_conv_kernel
            # Qwen3.6 在 q/k/v 上加深度可分离短卷积，增强局部性
            ch = H * D
            self.conv = nn.Conv1d(ch, ch, kernel_size=k, bias=False,
                                  groups=ch, padding=k - 1)
        else:
            self.conv = None

    # ------------------------------------------------------------------ #
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.n_heads, self.head_dim, self.head_dim)

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        def proj(linear: nn.Linear) -> torch.Tensor:
            return linear(x).view(B, T, H, D).transpose(1, 2)      # [B,H,T,D]

        q, k, v = proj(self.q_proj), proj(self.k_proj), proj(self.v_proj)
        if self.conv is not None:
            kk = self.cfg.gdn_conv_kernel
            def depthwise(t: torch.Tensor) -> torch.Tensor:
                y = self.conv(t.reshape(B, H * D, T)).reshape(B, H, D, T)
                return y[..., :T].transpose(2, 3)                  # 去掉右 padding
            q, k, v = depthwise(q.transpose(1, 2)), depthwise(k.transpose(1, 2)), depthwise(v.transpose(1, 2))
        # L2 归一化 + 尺度：GDN 原文对 q/k 做归一化以稳定 delta rule
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)       # [B,H,T]
        alpha = torch.sigmoid(self.a_proj(x)).transpose(1, 2)      # [B,H,T]
        return q, k, v, beta, alpha

    def _outputs(self, o: torch.Tensor) -> torch.Tensor:
        B, H, T, D = o.shape
        return self.o_proj(self.norm(o.transpose(1, 2)).reshape(B, T, H * D))

    def _new_state(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(ref.shape[0], *self.state_shape)

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="gdn") -> torch.Tensor:
        q, k, v, beta, alpha = self._project(x)
        B, H, T, D = q.shape
        if rope is not None:
            q, k = rope(q, k, positions)

        decode = (cache is not None and T == 1)
        if decode:
            state = cache.states.get(self.layer_idx) if cache is not None else None
            if state is None:
                state = self._new_state(q)
            out, state = self.forward_recurrent(q, k, v, beta, alpha, state)
            if cache is not None:
                cache.states[self.layer_idx] = state
        elif self.chunk_size > 0:
            state = cache.states.get(self.layer_idx) if cache is not None else None
            if state is None:
                state = self._new_state(q)
            out, state = self.forward_chunk(q, k, v, beta, alpha, state)
            if cache is not None:
                cache.states[self.layer_idx] = state
        else:
            out, state = self.forward_sequential(q, k, v, beta, alpha)
        return self._outputs(out)

    # ------------------------------------------------------------------ #
    #  形态 1：逐步递推（参考实现）
    # ------------------------------------------------------------------ #
    def forward_sequential(self, q, k, v, beta, alpha,
                           state: Optional[torch.Tensor] = None):
        """逐 token 递推：``S_t = α·S_{t-1} + β(v - S_{t-1}k)⊗k``。

        :param q/k/v: [B, H, T, D]
        :param beta/alpha: [B, H, T]
        """
        B, H, T, D = q.shape
        dev, dt = q.device, q.dtype
        S = self._new_state(q) if state is None else state.clone()
        outs = torch.zeros(B, H, T, D, dtype=dt, device=dev)
        for t in range(T):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]          # [B,H,D]
            at = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
            bt = beta[:, :, t].unsqueeze(-1)                         # [B,H,1]
            pred = torch.einsum("bhde,bhe->bhd", S, kt)              # S k → [B,H,D]
            err = vt - pred                                          # delta rule 的"误差"
            S = at * S + (bt * err).unsqueeze(-1) * kt.unsqueeze(-2)
            outs[:, :, t] = torch.einsum("bhde,bhe->bhd", S, qt)
        return outs, S

    # ------------------------------------------------------------------ #
    #  形态 2：分块扫描（数值稳定版）
    # ------------------------------------------------------------------ #
    def forward_chunk(self, q, k, v, beta, alpha,
                      state: Optional[torch.Tensor] = None):
        """分块扫描。

        与逐步递推**数学等价**，区别只有两点：

        1. 衰减连乘 ``Πα`` 在每个块内重新开始 → 长序列不会因 ``α^L`` 下溢；
        2. 块与块之间只传递固定大小的状态 S，这正是真实 kernel 的数据流
           （块内还需要 UT/WY 变换做真正的并行，本项目留作后续：见 docs/12）。
        """
        B, H, T, D = q.shape
        C = self.chunk_size
        S = self._new_state(q) if state is None else state.clone()
        outs = torch.zeros(B, H, T, D, dtype=q.dtype, device=q.device)
        for s in range(0, T, C):
            e = min(s + C, T)
            o, S = self.forward_sequential(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e],
                                           beta[:, :, s:e], alpha[:, :, s:e], S)
            outs[:, :, s:e] = o
        return outs, S

    # ------------------------------------------------------------------ #
    #  形态 3：单步 decode
    # ------------------------------------------------------------------ #
    def forward_recurrent(self, q, k, v, beta, alpha, state: torch.Tensor):
        """decode 一步：O(D²)，与上下文长度无关。"""
        qt, kt, vt = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        at = alpha[:, :, 0].unsqueeze(-1).unsqueeze(-1)        # [B,H,1,1]
        bt = beta[:, :, 0].unsqueeze(-1)                       # [B,H,1]
        pred = torch.einsum("bhde,bhe->bhd", state, kt)        # S k
        err = vt - pred
        S = at * state + (bt * err).unsqueeze(-1) * kt.unsqueeze(-2)
        out = torch.einsum("bhde,bhe->bhd", S, qt).unsqueeze(2)
        return out, S


# =========================================================================== #
#  Mamba-2 / SSD
# =========================================================================== #
class Mamba2Mixer(nn.Module):
    """Mamba-2 的结构化状态空间对偶（SSD）。

    状态更新（逐通道标量衰减，无 delta rule）::

        a_t = exp(-softplus(dt_t) · A)          # [H]，标量/头
        h_t = a_t · h_{t-1} + b_t ⊗ x_t         # h ∈ R^{H,N,P}
        y_t = Σ_n c_t[n] · h_t[n, :]            # [H,P]

    分块并行形式的推导（这也是 SSD 比 GDN 好写的原因——衰减是标量，连乘可闭式）::

        h_i = (Π_{t≤i} a_t) h_0 + Σ_{j≤i} (Π_{t=j+1}^{i} a_t) · b_j ⊗ x_j
        y_i = (Π_{t≤i} a_t)(c_iᵀ h_0) + Σ_{j≤i} (Π_{t=j+1}^{i} a_t)(c_i·b_j) x_j
                                         └── 这就是一个带衰减掩码的矩阵乘 ──┘

    于是块内**完全可以用两次 GEMM 算完**，不需要任何串行循环。
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.ssm_n_heads or cfg.n_heads
        self.head_dim = cfg.dim // self.n_heads
        self.state_size = cfg.ssm_state_size
        self.chunk_size = cfg.ssm_chunk_size
        H, P, N, C = self.n_heads, self.head_dim, self.state_size, cfg.dim

        self.in_dim = H * P                 # x
        self.in_proj = nn.Linear(C, 2 * H * P + 2 * H * N + H, bias=False)
        self.out_proj = nn.Linear(H * P, C, bias=False)
        self.norm = RMSNorm(P)

        if cfg.ssm_conv_kernel > 0:
            k = cfg.ssm_conv_kernel
            ch = H * P + 2 * H * N
            self.conv = nn.Conv1d(ch, ch, kernel_size=k, bias=False, groups=ch, padding=k - 1)
        else:
            self.conv = None

        # A > 0：越大遗忘越快（对数参数化保证正定）
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 16.0, H)))
        self.dt_bias = nn.Parameter(torch.zeros(H) - 2.0)

    # ------------------------------------------------------------------ #
    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.n_heads, self.state_size, self.head_dim)

    def _new_state(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(ref.shape[0], *self.state_shape)

    def _split(self, x: torch.Tensor, conv: bool = True):
        B, T, _ = x.shape
        H, P, N = self.n_heads, self.head_dim, self.state_size
        z = self.in_proj(x)                                      # [B,T,2HP+2HN+H]
        xi, bi, ci, dti, gate = torch.split(
            z, [H * P, H * N, H * N, H, H * P], dim=-1)
        if self.conv is not None and conv:
            k = self.cfg.ssm_conv_kernel
            cat = torch.cat([xi, bi, ci], dim=-1).transpose(1, 2)      # [B,ch,T]
            y = self.conv(cat)[..., :T].transpose(1, 2)
            xi, bi, ci = torch.split(y, [H * P, H * N, H * N], dim=-1)
        dt = F.softplus(dti + self.dt_bias)                      # [B,T,H] > 0
        A = F.softplus(self.A_log)                               # [H]
        a = torch.exp(-dt * A).transpose(1, 2)                   # [B,H,T] ∈ (0,1)
        x_ = xi.view(B, T, H, P).transpose(1, 2)                 # [B,H,T,P]
        b_ = bi.view(B, T, H, N).transpose(1, 2)                 # [B,H,T,N]
        c_ = ci.view(B, T, H, N).transpose(1, 2)
        g_ = gate.view(B, T, H, P).transpose(1, 2)
        return x_, b_, c_, a, g_

    # ------------------------------------------------------------------ #
    def forward(self, x, positions=None, rope=None, cache=None,
                attn_mask=None, is_causal=True, layer_type="mamba2") -> torch.Tensor:
        x_, b_, c_, a, g_ = self._split(x, conv=True)
        B, H, T, P = x_.shape
        decode = (cache is not None and T == 1)
        state = cache.states.get(self.layer_idx) if cache is not None else None
        if state is None:
            state = self._new_state(x)
        if decode:
            y, state = self.forward_recurrent(x_, b_, c_, a, state)
        else:
            y, state = self.forward_chunk(x_, b_, c_, a, state)
        if cache is not None:
            cache.states[self.layer_idx] = state
        y = self.norm((y * F.silu(g_)).transpose(1, 2)).reshape(B, T, H * P)
        return self.out_proj(y)

    # ------------------------------------------------------------------ #
    def forward_sequential(self, x_, b_, c_, a, state=None):
        """逐步递推参考实现。x_:[B,H,T,P]  b_,c_:[B,H,T,N]  a:[B,H,T]"""
        B, H, T, P = x_.shape
        N = self.state_size
        h = self._new_state(x_) if state is None else state.clone()
        ys = torch.zeros(B, H, T, P, dtype=x_.dtype, device=x_.device)
        for t in range(T):
            at = a[:, :, t].unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
            h = at * h + b_[:, :, t].unsqueeze(-1) * x_[:, :, t].unsqueeze(-2)
            ys[:, :, t] = torch.einsum("bhn,bhnp->bhp", c_[:, :, t], h)
        return ys, h

    def forward_chunk(self, x_, b_, c_, a, state=None):
        """分块并行：块内两次 GEMM，块间只传状态。"""
        B, H, T, P = x_.shape
        N, C = self.state_size, self.chunk_size
        h = self._new_state(x_) if state is None else state.clone()
        ys = torch.zeros(B, H, T, P, dtype=x_.dtype, device=x_.device)
        la = torch.log(a.clamp(min=1e-6))                        # [B,H,T]
        for s in range(0, T, C):
            e = min(s + C, T)
            n = e - s
            xc, bc, cc = x_[:, :, s:e], b_[:, :, s:e], c_[:, :, s:e]
            # 块内累积衰减：cum[i] = Σ_{t≤i} log a_t  →  D[i,j] = exp(cum[i]-cum[j])
            cum = torch.cumsum(la[:, :, s:e], dim=-1)            # [B,H,n]
            # 1) 块内项：y[i] = Σ_{j≤i} exp(cum[i]-cum[j]) (c_i·b_j) x_j
            scores = torch.einsum("bhtn,bhsn->bhts", cc, bc)     # [B,H,n,n]
            # dec[t,s] = log Π_{m=s+1..t} a_m。
            # 注意 a<1 → log a<0 → cum **递减**，所以「s 是过去(s<=t)」对应 dec<=0；
            # dec>0 意味着 s>t（未来），必须屏蔽。
            dec = cum.unsqueeze(-1) - cum.unsqueeze(-2)          # [B,H,n,n]
            dec = torch.where(dec <= 0, dec, torch.full_like(dec, -1e4))
            scores = scores * torch.exp(dec)
            y_intra = torch.einsum("bhts,bhsp->bhtp", scores, xc)
            # 2) 来自块初状态 h 的项：y[i] = exp(cum[i]) · (c_iᵀ h)
            y_state = torch.exp(cum).unsqueeze(-1) * torch.einsum("bhtn,bhnp->bhtp", cc, h)
            ys[:, :, s:e] = y_intra + y_state
            # 3) 块末状态：h_n = exp(cum[-1])·h + Σ_j exp(cum[-1]-cum[j]) · b_j⊗x_j
            #    第二项是 [N,n] @ [n,P]，一次 GEMM 完成（外积累加的批量形式）
            dcol = torch.exp(cum[:, :, -1:] - cum)               # [B,H,n]
            bmat = (dcol.unsqueeze(-1) * bc).transpose(2, 3)     # [B,H,N,n]
            h = torch.exp(cum[:, :, -1]).unsqueeze(-1).unsqueeze(-1) * h \
                + torch.einsum("bhnj,bhjp->bhnp", bmat, xc)
        return ys, h

    def forward_recurrent(self, x_, b_, c_, a, state):
        at = a[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        h = at * state + b_[:, :, 0].unsqueeze(-1) * x_[:, :, 0].unsqueeze(-2)
        y = torch.einsum("bhn,bhnp->bhp", c_[:, :, 0], h).unsqueeze(2)
        return y, h


# =========================================================================== #
def build_sequence_mixer(cfg: ModelConfig, layer_idx: int, kind: str) -> nn.Module:
    """按层类型构造序列混合器。"""
    if kind == "gdn":
        return GatedDeltaNet(cfg, layer_idx)
    if kind == "mamba2":
        return Mamba2Mixer(cfg, layer_idx)
    raise ValueError(f"未知序列混合器: {kind}（可选 gdn/mamba2）")
