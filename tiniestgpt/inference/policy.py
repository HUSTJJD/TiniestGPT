"""调度策略层：SLO-aware 调度、优先级、公平性、成本追踪、语义路由。

引擎里的 ``scheduler.py`` 解决的是"怎么把请求拼成 batch"，
本模块解决的是"**先服务谁**"以及"**这次服务值不值**"——
2026 年生产部署里这一层往往比 kernel 更能决定 p99。

四件事：

1. **SLO-aware**：TTFT 与 TPOT 是两个完全不同性质的指标。
   预填充长、解码短的请求要优先 prefill；长生成请求则要保证不被饿死。
2. **优先级 + 公平性**：纯优先级会让低优先级请求无限排队（starvation），
   所以用"带老化（aging）的加权公平队列"。
3. **成本追踪**：每个请求的 token 数 / 排队时长 / 显存占用都要记账，
   否则"优化"是盲的。
4. **语义路由**：按请求特征（长度、语言、是否含代码、是否需要工具）
   把它分派到最合适的模型/实例（vLLM Semantic Router 的思路）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = ["SLOConfig", "RequestCost", "FairScheduler", "CostLedger",
           "SemanticRouter", "PolicyDecision"]


@dataclass
class SLOConfig:
    ttft_ms: float = 500.0          # 首 token 延迟目标
    tpot_ms: float = 30.0           # 每 token 生成时间目标
    max_wait_ms: float = 5000.0
    aging_boost: float = 0.5        # 每等待 1 秒，优先级提升多少
    prefill_bonus: float = 1.0      # 短 prefill 的加分（先清掉快的，降低平均等待）


@dataclass
class RequestCost:
    rid: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    enqueue_at: float = field(default_factory=time.time)
    start_at: float = 0.0
    priority: int = 0
    kv_bytes: int = 0

    @property
    def waited_ms(self) -> float:
        return (time.time() - self.enqueue_at) * 1000.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class PolicyDecision:
    rid: str
    weight: float
    reason: str


class FairScheduler:
    """带老化的加权公平队列：高优先级先服务，但等待越久权重越高。"""

    def __init__(self, cfg: Optional[SLOConfig] = None) -> None:
        self.cfg = cfg or SLOConfig()
        self.queue: Dict[str, RequestCost] = {}
        self.served = 0
        self.starved = 0

    def enqueue(self, req: RequestCost) -> None:
        self.queue[req.rid] = req

    def pop_next(self) -> Optional[RequestCost]:
        if not self.queue:
            return None
        cfg = self.cfg
        best: Optional[RequestCost] = None
        best_w = float("-inf")
        for r in self.queue.values():
            w = r.priority * 10.0
            w += (r.waited_ms / 1000.0) * cfg.aging_boost        # 老化
            if r.waited_ms > cfg.max_wait_ms:
                w += 1000.0                                       # 超时兜底，强制插队
                self.starved += 1
            # 短 request 优先（减少平均等待时间，类似 SJF）
            if r.prompt_tokens > 0:
                w += cfg.prefill_bonus * (256.0 / max(r.prompt_tokens, 1))
            if w > best_w:
                best_w, best = w, r
        if best is None:
            return None
        self.queue.pop(best.rid)
        best.start_at = time.time()
        self.served += 1
        return best

    def report(self) -> str:
        return (f"FairScheduler: 排队 {len(self.queue)}，已服务 {self.served}，"
                f"触发防饿死 {self.starved} 次")


class CostLedger:
    """把"每个请求花了多少"记下来——没有这个，成本优化就是猜。"""

    def __init__(self, price_per_1k_prompt: float = 0.001,
                 price_per_1k_output: float = 0.002) -> None:
        self.p_in, self.p_out = price_per_1k_prompt, price_per_1k_output
        self.records: List[RequestCost] = []

    def record(self, r: RequestCost) -> float:
        self.records.append(r)
        return self.cost_of(r)

    def cost_of(self, r: RequestCost) -> float:
        return (r.prompt_tokens / 1000.0 * self.p_in +
                r.output_tokens / 1000.0 * self.p_out)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {}
        n = len(self.records)
        return {
            "requests": float(n),
            "tokens": float(sum(r.total_tokens for r in self.records)),
            "cost": float(sum(self.cost_of(r) for r in self.records)),
            "avg_wait_ms": float(sum((r.start_at - r.enqueue_at) * 1000
                                     for r in self.records if r.start_at) / n),
            "avg_output_tokens": float(sum(r.output_tokens for r in self.records) / n),
        }


class SemanticRouter:
    """按请求特征把它路由到最合适的后端。

    规则是**显式可解释**的（不是另一个模型），因为 2026 的生产实践
    更相信"可解释 + 可覆盖"，而不是再叠一层不透明的路由器。
    """

    def __init__(self, routes: Optional[Dict[str, str]] = None) -> None:
        self.routes = routes or {
            "code": "coder", "math": "reasoner", "chat": "general", "long": "longctx",
        }
        self.hits: Dict[str, int] = {}

    def classify(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("def ", "```", "import ", "function", "class ")):
            return "code"
        if any(k in p for k in ("求证", "证明", "solve", "integral", "=", "计算")):
            return "math"
        if len(prompt) > 4000:
            return "long"
        return "chat"

    def route(self, prompt: str) -> PolicyDecision:
        kind = self.classify(prompt)
        self.hits[kind] = self.hits.get(kind, 0) + 1
        return PolicyDecision(rid=self.routes.get(kind, "general"), weight=1.0,
                              reason=f"classified={kind}")

    def report(self) -> str:
        return f"SemanticRouter: {self.hits}"
