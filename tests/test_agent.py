"""Agent 框架测试：工具协议、ReAct 循环、上下文压缩、记忆、多智能体。"""

import torch

from tiniestgpt.agent import (Agent, AgentConfig, AgentRole, Blackboard, ContextConfig,
                              ContextManager, EchoBackend, MemoryManager, MultiAgentSystem,
                              Role, Supervisor, Tracer, VectorMemory)
from tiniestgpt.agent.builtin_tools import build_default_registry, calculator
from tiniestgpt.agent.planner import PlanExecutePlanner, ReActPlanner
from tiniestgpt.agent.tools import ToolRegistry, parse_tool_calls
from tiniestgpt.agent.types import Message


def test_tool_schema_from_type_hints():
    reg = ToolRegistry()

    def add(a: int, b: float = 1.0, note: str = "") -> str:
        """把两个数相加。

        :param a: 第一个数
        :param b: 第二个数
        """
        return str(a + b)

    spec = reg.register_fn(add)
    schema = spec.parameters
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "number"
    assert schema["required"] == ["a"]
    assert "把两个数相加" in spec.description


def test_tool_type_coercion():
    reg = build_default_registry()
    res = reg.invoke("calculator", {"expression": "2**10"}, call_id="c1")
    assert res.content == "1024"
    assert not res.is_error


def test_tool_error_returned_not_raised():
    reg = build_default_registry()
    res = reg.invoke("nonexistent_tool", {}, call_id="x") if False else None
    # 未注册工具应抛出 KeyError（由 runtime 捕获并转成错误结果）
    try:
        reg.invoke("nonexistent_tool", {})
        assert False, "应当抛出 KeyError"
    except KeyError:
        pass
    res = reg.invoke("read_file", {"path": "../../etc/passwd"})
    assert res.is_error or "不存在" in res.content or "越界" in res.content


def test_parse_tool_calls_formats():
    from tiniestgpt.agent.types import ToolCall

    calls = parse_tool_calls('```json\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n```')
    assert len(calls) == 1 and calls[0].name == "calculator"
    calls = parse_tool_calls('<tool_call>{"name": "think", "arguments": {"text": "hi"}}</tool_call>')
    assert len(calls) == 1 and calls[0].name == "think"


def test_react_agent_end_to_end():
    backend = EchoBackend(replies=[
        'Thought: 需要计算\nAction: calculator\nAction Input: {"expression": "2**10 + 3*(4+5)"}',
        "Final Answer: 结果是 1051",
    ])
    agent = Agent(backend, build_default_registry())
    r = agent.run("计算 2**10 + 3*(4+5)")
    assert r.success
    assert "1051" in r.answer
    assert r.tool_calls == 1
    assert r.steps == 2


def test_agent_stops_at_max_steps():
    # 后端永远返回工具调用 → 应该被 max_steps 截断而不是死循环
    backend = EchoBackend(replies=['Action: think\nAction Input: {"text": "loop"}' for _ in range(20)])
    agent = Agent(backend, build_default_registry(),
                  cfg=AgentConfig(max_steps=3, verbose=False))
    r = agent.run("无限循环任务")
    assert not r.success
    assert r.steps == 3


def test_react_planner_parsing():
    p = ReActPlanner()
    s = p.parse("Thought: 我在思考\nAction: calculator\nAction Input: {\"expression\": \"1+1\"}")
    assert s.thought == "我在思考"
    assert s.tool_calls[0].name == "calculator"
    assert s.tool_calls[0].arguments == {"expression": "1+1"}
    s2 = p.parse("Final Answer: 42")
    assert s2.is_final and s2.final_answer == "42"


def test_plan_execute_planner():
    p = PlanExecutePlanner()
    plan = p.parse_plan('好的：["第一步", "第二步", "第三步"]，开始吧')
    assert plan == ["第一步", "第二步", "第三步"]


def test_context_compaction():
    cfg = ContextConfig(max_context_tokens=200, reserve_for_output=20, keep_recent=2)
    cm = ContextManager(cfg)
    msgs = [Message(role=Role.USER, content="x" * 300) for _ in range(6)]
    assert cm.should_compact(msgs)
    out = cm.compact(msgs)
    assert len(out) <= len(msgs)
    assert any("历史摘要" in m.content for m in out)
    assert cm.stats["compactions"] == 1


def test_context_truncates_tool_result():
    cm = ContextManager(ContextConfig(max_tool_result_chars=100))
    long_text = "a" * 1000
    out = cm.truncate_tool_result(long_text)
    assert len(out) < 300 and "截断" in out


def test_vector_memory_recall():
    mem = VectorMemory(dim=64)
    mem.add("巴黎是法国的首都", kind="fact")
    mem.add("水在 100 摄氏度沸腾", kind="fact")
    mem.add("Python 是一种编程语言", kind="fact")
    hits = mem.search("法国的首都是哪里", k=1)
    assert "巴黎" in hits[0].text


def test_memory_manager_overflow_to_summary():
    mm = MemoryManager(working_capacity=3)
    for i in range(10):
        mm.observe(f"事件 {i}")
    assert len(mm.working.items) <= 3
    assert len(mm.summary.items) >= 1


def test_multi_agent_supervisor():
    backend = EchoBackend(replies=["Final Answer: 子任务完成"])
    system = MultiAgentSystem(backend=backend, verbose=False)
    system.register(AgentRole(name="researcher", description="检索信息",
                              tools=build_default_registry()))
    system.register(AgentRole(name="coder", description="写代码", tools=build_default_registry()))
    sup: Supervisor = system.supervisor()
    r = sup.run("帮我查一下资料")
    assert r.success


def test_blackboard():
    bb = Blackboard()
    bb.write("k", "v", author="a")
    assert bb.read("k") == "v"
    assert len(bb.log) == 1


def test_tracer_records_events(tmp_path):
    tracer = Tracer(path=str(tmp_path / "trace.jsonl"), run_id="r1")
    tracer.emit(__import__("tiniestgpt.agent.types", fromlist=["EventType"]).EventType.RUN_START, task="t")
    tracer.close()
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "RUN_START" in lines[0] or "run_start" in lines[0]


def test_plan_execute_runtime_runs():
    backend = EchoBackend(replies=[
        '["先计算", "再回答"]',
        'Action: calculator\nAction Input: {"expression": "6*7"}',
        "Final Answer: 42",
        "最终答案是 42",
    ])
    agent = Agent(backend, build_default_registry(),
                  cfg=AgentConfig(planner="plan_execute", verbose=False))
    r = agent.run_plan_execute("6 乘 7 是多少")
    assert r.steps >= 1
