"""Prometheus 指标导出（纯文本 exposition format，零依赖）。

生产推理服务必须有可观测性，否则"用户说慢"这件事无法定位。
核心指标就那几个（AIInfraGuide 路线 3.6）：

* ``tiniestgpt_requests_total``       —— 累计请求数
* ``tiniestgpt_requests_running``     —— 正在处理（等待 + 生成中）
* ``tiniestgpt_prompt_tokens_total`` / ``generation_tokens_total``
* ``tiniestgpt_ttft_seconds``         —— 首 token 延迟（histogram）
* ``tiniestgpt_tpot_seconds``         —— 每 token 延迟（histogram）
* ``tiniestgpt_kv_cache_usage_ratio`` —— KV Cache 占用率（**OOM 预警的关键指标**）
* ``tiniestgpt_preemptions_total``    —— 抢占次数（频繁抢占 = 显存不够）
* ``tiniestgpt_step_seconds``         —— 单步调度 + 前向耗时

刻意**不引入 prometheus_client**：文本格式很简单，少一个依赖就少一份版本负担。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

__all__ = ["MetricsRegistry", "HISTOGRAM_BUCKETS", "default_registry"]

# 推理场景常用的桶边界（秒）：从 1ms 到 10s
HISTOGRAM_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """极简指标注册表：counter / gauge / histogram 三类。"""

    def __init__(self, prefix: str = "tiniestgpt",
                 buckets=HISTOGRAM_BUCKETS) -> None:
        self.prefix = prefix
        self.buckets = tuple(buckets)
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        # name -> {"sum": float, "count": int, "buckets": {le: count}}
        self.histograms: Dict[str, Dict] = {}
        self._help: Dict[str, str] = {}
        self._types: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def _fq(self, name: str) -> str:
        return name if name.startswith(self.prefix) else f"{self.prefix}_{name}"

    def counter(self, name: str, value: float = 1.0, help_text: str = "") -> None:
        k = self._fq(name)
        self.counters[k] = self.counters.get(k, 0.0) + value
        self._help.setdefault(k, help_text)
        self._types[k] = "counter"

    def gauge(self, name: str, value: float, help_text: str = "") -> None:
        self.gauges[self._fq(name)] = float(value)
        self._help.setdefault(self._fq(name), help_text)
        self._types[self._fq(name)] = "gauge"

    def observe(self, name: str, value: float, help_text: str = "") -> None:
        k = self._fq(name)
        h = self.histograms.setdefault(
            k, {"sum": 0.0, "count": 0, "buckets": {le: 0 for le in self.buckets}})
        h["sum"] += float(value)
        h["count"] += 1
        for le in self.buckets:
            if value <= le:
                h["buckets"][le] += 1
        self._help.setdefault(k, help_text)
        self._types[k] = "histogram"

    # ------------------------------------------------------------------ #
    def observe_ttft(self, seconds: float) -> None:
        self.observe("ttft_seconds", seconds, "首 token 延迟（秒）")

    def observe_tpot(self, seconds: float) -> None:
        self.observe("tpot_seconds", seconds, "每输出 token 的延迟（秒）")

    def observe_step(self, seconds: float) -> None:
        self.observe("step_seconds", seconds, "单步调度 + 前向耗时（秒）")

    # ------------------------------------------------------------------ #
    def update_from_engine(self, engine, extra: Optional[Dict[str, float]] = None) -> None:
        """从引擎状态刷新 gauge（每次 /metrics 被拉取时调用）。"""
        try:
            st = engine.stats()
        except Exception:  # pragma: no cover
            st = {}
        for key, name in (("block_usage", "kv_cache_usage_ratio"),
                          ("preempted", "preemptions_total"),
                          ("steps", "steps_total")):
            if key in st:
                if name.endswith("_total"):
                    self.counters[self._fq(name)] = float(st[key])
                else:
                    self.gauge(name, float(st[key]), "KV Cache 块占用率")
        running = len(getattr(engine.scheduler, "running", []))
        waiting = len(getattr(engine.scheduler, "waiting", []))
        self.gauge("requests_running", running, "正在生成的请求数")
        self.gauge("requests_waiting", waiting, "排队中的请求数")
        free = st.get("free_blocks")
        if free is not None:
            self.gauge("kv_cache_free_blocks", float(free), "空闲 KV 块数量")
        for k, v in (extra or {}).items():
            self.gauge(k, v)

    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """输出 Prometheus 文本格式。"""
        lines: List[str] = []
        for name, value in sorted(self.counters.items()):
            lines.append(f"# HELP {name} {_escape(self._help.get(name, ''))}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, value in sorted(self.gauges.items()):
            lines.append(f"# HELP {name} {_escape(self._help.get(name, ''))}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        for name, h in sorted(self.histograms.items()):
            help_text = _escape(self._help.get(name, ""))
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} histogram")
            for le in self.buckets:
                lines.append(f'{name}_bucket{{le="{le}"}} {h["buckets"][le]}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {h["count"]}')
            lines.append(f"{name}_sum {h['sum']}")
            lines.append(f"{name}_count {h['count']}")
        lines.append(f"# HELP {self.prefix}_scrape_timestamp_seconds 上次采集时间")
        lines.append(f"# TYPE {self.prefix}_scrape_timestamp_seconds gauge")
        lines.append(f"{self.prefix}_scrape_timestamp_seconds {time.time()}")
        return "\n".join(lines) + "\n"


default_registry = MetricsRegistry()
