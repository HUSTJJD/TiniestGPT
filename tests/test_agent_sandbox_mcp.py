"""Agent 侧两件 2026 必备件：真沙箱 + MCP 协议。"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from tiniestgpt.agent.mcp import MCPClient, MCPServer, MCPToolBridge, echo_server_script
from tiniestgpt.agent.sandbox import Sandbox, SandboxConfig, run_python
from tiniestgpt.agent.tools import ToolRegistry

# 沙箱要起子进程 + 等超时，Windows 上 spawn 开销较大，给足时间
SANDBOX_TIMEOUT = 20.0


# --------------------------------------------------------------------------- #
#  沙箱
# --------------------------------------------------------------------------- #
def test_sandbox_runs_normal_code():
    r = run_python("print(6*7)", SandboxConfig(timeout=SANDBOX_TIMEOUT))
    assert r.ok
    assert "42" in r.stdout


def test_sandbox_kills_infinite_loop():
    """`while True: pass` 必须被超时杀掉，而不是把测试进程挂死。"""
    r = run_python("while True:\n    pass", SandboxConfig(timeout=3.0))
    assert not r.ok
    assert "超时" in r.error or "资源" in r.error


def test_sandbox_truncates_huge_output():
    r = run_python('print("x" * 50000)', SandboxConfig(timeout=SANDBOX_TIMEOUT,
                                                       max_output_chars=1000))
    assert r.truncated
    assert len(r.stdout) <= 1000


def test_sandbox_persistent_workdir():
    """长程 Agent 的前提：上一步写的文件，下一步还在。"""
    with tempfile.TemporaryDirectory() as d:
        sb = Sandbox(SandboxConfig(timeout=SANDBOX_TIMEOUT, workdir=d))
        sb.write_file("data.txt", "hello")
        assert sb.run("print(open('data.txt').read())").stdout.strip().endswith("hello")
        assert sb.calls == 1


def test_sandbox_blocks_network():
    r = run_python(
        "import socket\ntry:\n    socket.socket()\n    print('NET_OK')\n"
        "except Exception as e:\n    print('NET_BLOCKED')",
        SandboxConfig(timeout=SANDBOX_TIMEOUT, allow_network=False))
    assert "NET_BLOCKED" in r.stdout or not r.ok


def test_builtin_python_repl_uses_sandbox():
    from tiniestgpt.agent import builtin_tools as bt

    out = bt.python_repl("print(1+1)", timeout=5.0, enabled=True)
    assert "2" in out
    assert "禁用" in bt.python_repl("print(1)", enabled=False)


# --------------------------------------------------------------------------- #
#  MCP
# --------------------------------------------------------------------------- #
def _write_echo_server(path: Path) -> Path:
    p = path / "mcp_echo_server.py"
    p.write_text(echo_server_script(), encoding="utf-8")
    return p


def test_mcp_handshake_and_tool_call():
    """不装任何依赖，用内置 echo server 跑通 initialize → tools/list → tools/call。"""
    with tempfile.TemporaryDirectory() as d:
        srv = _write_echo_server(Path(d))
        client = MCPClient(MCPServer(name="echo", command=[sys.executable, str(srv)],
                                     timeout=SANDBOX_TIMEOUT))
        try:
            res = client.initialize()
            assert "result" in res, f"initialize 失败: {res}"
            tools = client.list_tools()
            assert any(t["name"] == "echo" for t in tools)
            out = client.call_tool("echo", {"text": "hi"})
            assert "hi" in out
        finally:
            client.close()


def test_mcp_bridge_registers_tools_into_registry():
    with tempfile.TemporaryDirectory() as d:
        srv = _write_echo_server(Path(d))
        reg = ToolRegistry()
        bridge = MCPToolBridge(reg)
        try:
            names = bridge.add_server(MCPServer(name="echo",
                                                command=[sys.executable, str(srv)],
                                                timeout=SANDBOX_TIMEOUT))
            assert names == ["echo__echo"]
            assert "echo__echo" in reg
            res = reg.invoke("echo__echo", {"text": "hello"})
            assert "hello" in res.content and not res.is_error
        finally:
            bridge.close()
