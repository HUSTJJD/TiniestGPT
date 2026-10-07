# Agentic 框架：让模型"能动手、会反省、可协作"

## 1. 整体结构

```
Backend（大脑）  ←→  Agent（闭环）
                      ├─ Planner（怎么想）
                      ├─ ToolRegistry（能做什么）
                      ├─ Memory（记得什么）
                      ├─ ContextManager（装得下多少）
                      └─ Tracer（发生了什么）
```

## 2. 工具协议（`tools.py`）

**单一事实来源**：一份 Python 类型标注 + docstring 同时生成
JSON Schema（给模型看）与运行时校验（给人看），避免"文档与实现不一致"。

```python
@tool(name="my_search", description="搜索内部知识库")
def my_search(query: str, k: int = 3) -> str:
    """按关键词检索。

    :param query: 检索关键词
    :param k: 返回条数
    """
    ...
```

- 错误**不抛出**，而是把错误文本回灌给模型 —— "让模型自己修"是鲁棒性的核心；
- `dangerous=True` 标记需要沙箱/二次确认的工具（`python_repl` 默认禁用）；
- 小模型往往吐不出严格 JSON，因此 `parse_tool_calls` 做尽力解析：
  支持 `<tool_call>` 标签、```json 代码块、裸 JSON 三种形态。

## 3. 规划范式（`planner.py`）

| 范式 | 适用 | 特点 |
|---|---|---|
| **ReAct** | 通用工具任务 | Thought→Action→Observation，最省 token |
| **Plan-and-Execute** | 多步依赖的长任务 | 规划与执行可各用不同模型（降本） |
| **Reflexion** | 易反复犯同类错误 | 失败后写反思进记忆再重试 |

Planner 只负责"生成提示词"与"解析输出"，执行与记忆交给 runtime —— 三者可自由组合。

## 4. 记忆（`memory.py`）

四层，对应不同的代价/保真度权衡：

| 层 | 内容 | 代价 |
|---|---|---|
| 工作记忆 | 当前对话原始消息 | 占 context |
| 摘要记忆 | 滚出窗口的内容压缩 | 有损但便宜 |
| 向量记忆 | 长期知识，语义检索 | 容量大 |
| 情景记忆 | 任务→轨迹→结果 | 用于经验复用（Reflexion） |

向量检索用**特征哈希 + TF-IDF 权重**实现（零依赖、可离线），
同时保留 `embed_fn` 接口，方便换成真正的句向量模型。

## 5. 上下文管理（`context.py`）

长任务 Agent 的第一工程难题。本项目组合了：

1. 滑动窗口（保留最近 N 条）；
2. 摘要压缩（早期内容用 LLM 压成一段）；
3. 检索注入（只取回相关历史，由 `VectorMemory` 提供）；
4. 工具结果截断（几千 token 先截断再入库）；
5. 系统提示常驻（永不被压缩）。

## 6. 多智能体（`multi_agent.py`）

为什么不是一个全能 Agent：

- **上下文隔离**：子 Agent 只看自己那部分；
- **工具隔离**：搜索 Agent 与代码 Agent 用不同工具集，减少误用；
- **并行**：独立子任务同时跑；
- **可替换**：强模型做规划、小模型做执行。

三种拓扑都实现了：**Supervisor**（中心化，最易调试）、
**Blackboard**（共享黑板，协作式求解）、**Handoff**（交接，客服/路由场景）。

## 7. 可观测性（`observability.py`）

Agent 的失败往往"不可复现"（LLM 有随机性、工具会超时、上下文会漂移），
因此必须把**每次运行完整落盘**：`Tracer` 写 JSONL 事件流，`Metrics` 聚合计数器与耗时。

```bash
python -m tiniestgpt.cli agent --backend local --checkpoint out/tiny/last.pt \
    --task "计算 2**10 + 3*(4+5)" --trace out/trace.jsonl
```

## 8. 不依赖模型也能验证逻辑

```python
from tiniestgpt.agent import Agent, EchoBackend
from tiniestgpt.agent.builtin_tools import build_default_registry

backend = EchoBackend(replies=[
    'Thought: 需要计算\nAction: calculator\nAction Input: {"expression": "6*7"}',
    "Final Answer: 42",
])
print(Agent(backend, build_default_registry()).run("6 乘 7 是多少").answer)
```

这条路径把"Agent 逻辑正确性"与"模型能力"解耦——
**先保证框架对，再换更好的模型**。
