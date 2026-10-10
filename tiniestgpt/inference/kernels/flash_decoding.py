"""Flash-Decoding / split-KV：长上下文 decode 的关键 kernel 思路。

decode 时 Q 只有 1 个 token，但要读 **L 个**历史 K/V。
如果只用 1 个 CUDA block 处理，整个 attention 就是串行的——
GPU 上成千上万个 SM 只有一个在干活。

split-KV 的做法：**沿 KV 维度切分**，每个 block 算一段的部分结果
（各自的 running max / sum / accumulator），最后再 **reduce 合并**。

这正是 ``context_parallel.OnlineSoftmaxState`` 用的同一个数学工具——
因为 softmax 可以分块归约，只要把 (max, sum, acc) 三元组一起传。

适用场景与限制：

* 上下文越长、batch 越小，收益越大（SM 才有东西可并行）；
* batch 很大时，单靠 batch 并行就够了，split-KV 反而增加 reduce 开销；
* 与 GQA/MQA 配合时要小心：split 的是 **KV 位置**，不是 head。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

__all__ = ["SplitKVConfig", "split_kv_attention", "merge_partials",
           "recommend_split", "flash_decoding"]


@dataclass
class SplitKVConfig:
    n_split: int = 4
    min_chunk: int = 256          # 每段最少多少个 KV 位置（太碎会增加 reduce 开销）
    causal: bool = True


def recommend_split(kv_len: int, n_sm: int = 68, batch: int = 1,
                    min_chunk: int = 256) -> int:
    """按"SM 数 ÷ batch"估算切多少段最合适。

    目标是让 ``batch × n_split`` 略大于 SM 数，把 GPU 填满但不至于太碎。
    """
    if kv_len <= min_chunk:
        return 1
    target = max(n_sm // max(batch, 1), 1)
    return int(min(max(target, 1), kv_len // min_chunk))


def split_kv_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       cfg: SplitKVConfig, offset: int = 0
                       ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """沿 KV 维切成 n_split 段，各自算 (m, l, acc)。

    q:[B,H,T,D]  k/v:[B,S,H,D]（或 [B,H,S,D]，见 ``transposed`` 约定：本函数统一用 [B,S,H,D]）
    offset: q 的第一个位置在完整序列中的绝对下标（decode 时 T=1 且 offset>0）
    """
    B, S, H, D = k.shape
    T = q.shape[2] if q.dim() == 4 else q.shape[1]
    n = max(int(cfg.n_split), 1)
    n = min(n, max(S // cfg.min_chunk, 1))
    per = (S + n - 1) // n
    parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    scale = D ** 0.5

    for i in range(n):
        lo, hi = i * per, min((i + 1) * per, S)
        if hi <= lo:
            continue
        kc = k[:, lo:hi].transpose(1, 2)          # [B,H,seg,D]
        vc = v[:, lo:hi].transpose(1, 2)
        s = torch.matmul(q, kc.transpose(-1, -2)) / scale     # [B,H,T,seg]
        if cfg.causal:
            q_abs = offset + torch.arange(T, device=q.device).view(1, 1, -1, 1)
            k_abs = lo + torch.arange(hi - lo, device=q.device).view(1, 1, 1, -1)
            s = s.masked_fill(k_abs > q_abs, float("-inf"))
        m = s.amax(dim=-1)
        p = torch.exp(s - m.unsqueeze(-1))
        l = p.sum(dim=-1)
        acc = torch.matmul(p, vc)
        parts.append((m, l, acc))
    return parts


def merge_partials(parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                   ) -> torch.Tensor:
    """用统一的 max 基准合并各段的 (m, l, acc)——online softmax 的标准合并。"""
    if not parts:
        raise ValueError("没有可合并的分段")
    m_all = torch.stack([p[0] for p in parts], dim=0)
    m = m_all.amax(dim=0)
    l = torch.zeros_like(m)
    acc = torch.zeros_like(parts[0][2])
    for mi, li, ai in parts:
        corr = torch.exp(mi - m)
        l = l + li * corr
        acc = acc + ai * corr.unsqueeze(-1)
    return acc / l.clamp(min=1e-12).unsqueeze(-1)


def flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   cfg: Optional[SplitKVConfig] = None, offset: int = 0,
                   auto: bool = True) -> torch.Tensor:
    """对外入口。``auto=True`` 时按 KV 长度自动决定切分数。"""
    cfg = cfg or SplitKVConfig()
    S = k.shape[1]
    if auto:
        cfg = SplitKVConfig(n_split=recommend_split(S, min_chunk=cfg.min_chunk),
                            min_chunk=cfg.min_chunk, causal=cfg.causal)
    return merge_partials(split_kv_attention(q, k, v, cfg, offset))
