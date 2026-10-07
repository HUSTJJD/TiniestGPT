# 10 · 服务化、可观测与部署

> 对应 AIInfraGuide 路线 3.6（性能分析与 Benchmark）、3.7（优化选型决策树）、
> 以及"生产级服务特性"一节。代码在 `inference/metrics.py`、`inference/structured.py`、
> 根目录 `Dockerfile` / `docker-compose.yml`。

---

## 1. 六项指标的基准（可复现、可回归）

```bash
python benchmarks/serving_bench.py --num-requests 64 --concurrency 16
python benchmarks/serving_bench.py --checkpoint out/tiny/last.pt --tokenizer data/tokenizer.json

# 存基线 → 之后每次改动都拿来对比（可直接接 CI）
python benchmarks/serving_bench.py --save-baseline out/serving_baseline.json
python benchmarks/serving_bench.py --baseline out/serving_baseline.json --max-regression 0.05
```

| 指标 | 为什么必须有 |
|---|---|
| QPS | 容量规划 |
| TTFT P50 / P95 | 首 token 延迟，直接决定"快不快"的体感 |
| TPOT P50 / P95 | 每 token 延迟，决定"卡不卡" |
| 端到端 token/s | 与训练侧口径对齐的吞吐 |
| 显存峰值 | 能不能再提并发 |
| GPU 繁忙率 | 调度有没有把 GPU 喂饱（低 = 有 CPU/调度瓶颈） |
| **Goodput** | 满足 SLO 的请求占比 —— **raw QPS 高 ≠ 用户体验好** |

回归门禁规则（建议）：TPOT P95 退化 >5% 就 block merge；显存增长 >10% 需要附分析报告。

---

## 2. Prometheus 指标

```bash
curl http://localhost:8000/metrics
```

输出标准 Prometheus 文本格式（无第三方依赖）：

```
tiniestgpt_requests_running 3
tiniestgpt_requests_waiting 1
tiniestgpt_kv_cache_usage_ratio 0.42
tiniestgpt_preemptions_total 7.0
tiniestgpt_ttft_seconds_bucket{le="0.5"} 12
tiniestgpt_ttft_seconds_sum ...
```

排障速查：

| 现象 | 看哪个指标 | 含义 |
|---|---|---|
| 用户抱怨慢但 QPS 很高 | `ttft_seconds` P95 | 排队时间长，goodput 低 |
| 频繁卡顿 | `tpot_seconds` P95 | decode 被 prefill 拖慢 → 考虑 PD 解耦 |
| 突然 OOM | `kv_cache_usage_ratio` | 接近 1 → 并发上限或该扩容 |
| 吞吐抖动 | `preemptions_total` | 抢占频繁 = 显存不够，考虑 swap / 降并发 |

Grafana 面板直接读这些指标即可；`deploy/prometheus.yml` 已经写好抓取配置。

---

## 3. 结构化输出（约束解码）

```python
from tiniestgpt.inference.sampler import SamplingParams

schema = {"type": "object",
          "required": ["name"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
sid = engine.add_request(prompt, SamplingParams(max_tokens=64), json_schema=schema)
```

原理（见 `inference/structured.py`）：

1. 一个**增量 JSON 语法状态机**（`JsonPrefix`）知道"当前位置允许哪些字符"；
2. 每一步把词表里每个 token 喂给状态机的**副本**，走不通的 logits 置为 `-inf`；
3. JSON 一闭合就提前结束（不用等 `max_tokens`）。

> 这是 Agent 工具调用能不能稳定落地的关键：
> 靠 prompt 说"请输出 JSON"是**概率**，靠 logits masking 是**保证**。

与 vLLM 的差距：vLLM 用预编译 DFA + 前缀树把这一步压到微秒级，
并且支持完整 JSON Schema / 正则；这里是纯 Python 逐 token 模拟，
教学可读性优先。

---

## 4. Prefill / Decode 解耦实验

```bash
python benchmarks/pd_disaggregation.py --n-long 3 --n-short 4 --long-prompt 256
```

输出"混合部署 vs 纯 decode"的**干扰系数**，以及解耦的代价：

* 每请求要迁移的 KV 字节数
* 在 IB 带宽下的迁移耗时
* 盈亏平衡：单请求要生成多少 token，解耦才回本

结论（也是 DistServe 的核心论点）：**不是所有业务都该解耦** ——
短回答 + 短 prompt 的场景，迁移开销大于收益。

---

## 5. 部署

```bash
docker compose up -d --build
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

* 镜像基于 `nvidia/cuda:12.8.0-**devel**`：带 nvcc，
  于是 `tiniestgpt/kernels` 的手写 CUDA kernel 可以在容器内 JIT 编译；
* 权重与数据用 volume 挂载（`./out`、`./data`）；
* Prometheus 容器默认抓取 `tiniestgpt:8000/metrics`。

多卡训练：

```bash
bash scripts/train_torchrun.sh                                   # 2 卡
NPROC=4 CONFIG=recipes/pretrain_moe.yaml bash scripts/train_torchrun.sh
```

---

## 6. 优化选型决策树（速查）

| 症状 | 先看 | 处方 |
|---|---|---|
| **OOM / 并发上不去** | `kv_cache_usage_ratio` | 先解决显存：调小 `max_num_seqs`、开 KV 量化、开 swap |
| **TTFT 高** | `ttft_seconds` P95 | 长 prompt → Chunked Prefill；重复前缀 → Prefix Caching / Radix |
| **TPOT 高** | `tpot_seconds` P95 | KV 带宽 → FlashAttention / Triton 内核；并发不足 → 提高 `max_num_seqs` |
| **P95 毛刺** | `preemptions_total` | prefill 干扰 → Prefill/Decode 解耦；或开 swap 减少重算 |

顺序永远是：**先解决 OOM → 再优化 TTFT → 然后 TPOT/吞吐 → 最后尾延迟**。

---

## 7. 还差什么（与生产的差距清单）

| 能力 | 本项目 | 生产引擎 |
|---|---|---|
| 张量并行 | ❌ | vLLM / Megatron 完整支持 |
| 结构化输出 | JSON 语法级 | 完整 JSON Schema / 正则 + 预编译 DFA |
| 指标 | 文本格式 + 手动埋点 | 全套 + 自动埋点 + 分布式追踪 |
| 多 LoRA / 多模型 | ❌ | 支持 |
| sleep/wake、在线更新权重 | ❌ | 支持 |
| KV 跨节点迁移 | 有 CPU swap 通道 | RDMA / NIXL |
