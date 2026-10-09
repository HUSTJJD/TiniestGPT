"""MCP（Model Context Protocol）客户端。

2026 年 MCP 已经是**工具层的事实标准**（"AI 的 USB-C"）：
与其给每个 Agent 框架各写一套工具协议，不如让工具暴露成一个 MCP server，
任何支持 MCP 的 Agent 都能直接 ``tools/list`` + ``tools/call``。

本模块实现的是**最小可用的 MCP 客户端**，两个 transport 都支持：

* ``stdio`` —— 启动一个子进程，用换行分隔的 JSON-RPC 通信（本地工具的主流方式）；
* ``http``  —— 直接 POST 到 HTTP 端点（远程工具）。

协议要点（JSON-RPC 2.0）::

    → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
    ← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}
    → {"jsonrpc":"2.0","id":2,"method":"tools/list"}
    ← {"jsonrpc":"2.0","id":2,"result":{"tools":[{name, description, inputSchema}]}}
    → {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":..., "arguments":{...}}}
    ← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}]}}

同时提供 :class:`MCPToolBridge`：把 MCP server 的工具**直接注册进** :class:`ToolRegistry`，
于是 Agent 不用改一行代码就能调用生态里的工具。
"""

from __future__ import annotations

import inspect
import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["MCPServer", "MCPClient", "MCPToolBridge", "echo_server_script"]


# --------------------------------------------------------------------------- #
@dataclass
class MCPServer:
    """一个 MCP server 的连接描述。"""

    name: str
    command: Optional[List[str]] = None      # stdio 模式：启动命令
    url: Optional[str] = None                # http 模式：端点
    env: Dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0


# --------------------------------------------------------------------------- #
class MCPClient:
    """同步 MCP 客户端（教学实现：够用、可读、零依赖）。"""

    JSONRPC = "2.0"

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._tools: List[Dict[str, Any]] = []
        self._init = False

    # ------------------------------------------------------------------ #
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        req = {"jsonrpc": self.JSONRPC, "id": self._next_id(), "method": method}
        if params is not None:
            req["params"] = params
        if self.server.url:
            return self._send_http(req)
        return self._send_stdio(req)

    def _send_http(self, req: Dict) -> Dict:
        import urllib.request

        data = json.dumps(req).encode("utf-8")
        r = urllib.request.Request(self.server.url, data=data,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.server.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _send_stdio(self, req: Dict) -> Dict:
        if self.proc is None:
            import os

            # 必须强制 UTF-8：Windows 上子进程默认用本地代码页（GBK），
            # 中文的工具描述会被 Popen(encoding="utf-8") 解成乱码。
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            if self.server.env:
                env.update(self.server.env)
            self.proc = subprocess.Popen(
                self.server.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env,
                bufsize=1,
            )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        # 逐行读，跳过非 JSON 的噪声行（server 常会先打日志）
        deadline = time.time() + self.server.timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
        return {"error": {"code": -32000, "message": "MCP server 无响应（超时）"}}

    # ------------------------------------------------------------------ #
    def initialize(self) -> Dict[str, Any]:
        res = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "tiniestgpt", "version": "0.1"},
        })
        self._init = "error" not in res
        return res

    def list_tools(self) -> List[Dict[str, Any]]:
        if not self._init:
            self.initialize()
        res = self._send("tools/list")
        self._tools = (res.get("result") or {}).get("tools", [])
        return self._tools

    def call_tool(self, name: str, arguments: Optional[Dict] = None) -> str:
        if not self._tools:
            self.list_tools()
        res = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        if "error" in res:
            return f"[MCP error] {res['error'].get('message', res['error'])}"
        r = res.get("result") or {}
        parts = r.get("content") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join(t for t in texts if t) or json.dumps(r, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _make_caller(client: "MCPClient", tool_name: str, schema: Dict[str, Any]):
    """生成"调某个 MCP 工具"的闭包。

    三处细节：
    1. 用工厂函数而不是 ``lambda **kw, _n=...``（**语法不允许**关键字参数后再跟参数），
       也顺带避开循环里 lambda 的**迟绑定**陷阱（否则所有工具都去调最后一个名字）；
    2. 伪造 ``__signature__``，让 :class:`ToolRegistry` 既能生成正确的 JSON Schema，
       也能做运行时类型校验——否则 ``**kwargs`` 会被当成"必填参数 kw"；
    3. ``required`` 之外的参数给 ``None`` 默认值。
    """
    props = list((schema or {}).get("properties", {}).keys())
    required = set((schema or {}).get("required", []) or [])

    def call(*args, **kw):
        payload = dict(zip(props, args))
        payload.update(kw)
        return client.call_tool(tool_name, payload)

    call.__signature__ = inspect.Signature([          # type: ignore[attr-defined]
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=None if n not in required else inspect.Parameter.empty)
        for n in props
    ])
    call.__name__ = f"mcp_{tool_name}"
    call.__doc__ = f"调用 MCP 工具 {tool_name}"
    return call


# --------------------------------------------------------------------------- #
class MCPToolBridge:
    """把 MCP server 暴露的工具注册进本地 :class:`ToolRegistry`。

    用法::

        bridge = MCPToolBridge(registry)
        bridge.add_server(MCPServer(name="fs", command=["python", "fs_server.py"]))
        # 之后 Agent 就能像调用本地工具一样调用它们
    """

    def __init__(self, registry) -> None:
        self.registry = registry
        self.clients: Dict[str, MCPClient] = {}
        self.tool_owner: Dict[str, str] = {}

    def add_server(self, server: MCPServer) -> List[str]:
        client = MCPClient(server)
        self.clients[server.name] = client
        names: List[str] = []
        for t in client.list_tools():
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            name = f"{server.name}__{t['name']}"
            from .tools import ToolSpec

            spec = ToolSpec(
                name=name,
                description=f"[MCP:{server.name}] {t.get('description', '')}",
                parameters=schema,
                func=_make_caller(client, t["name"], schema),
            )
            self.registry.register(spec)
            self.tool_owner[name] = server.name
            names.append(name)
        return names

    def close(self) -> None:
        for c in self.clients.values():
            c.close()


# --------------------------------------------------------------------------- #
def echo_server_script() -> str:
    """返回一个**可运行的 MCP server 示例脚本**（用于离线验证协议）。

    真实场景请换成社区里已有的 server；这个脚本的意义是
    "不装任何依赖也能把 MCP 握手跑通"。
    """
    return '''"""MCP echo server：最小可运行示例（stdio transport）。"""
import json
import sys

TOOLS = [{
    "name": "echo",
    "description": "把输入原样返回（用于验证 MCP 链路）",
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
}]

def handle(req):
    m = req.get("method")
    i = req.get("id")
    if m == "initialize":
        return {"jsonrpc": "2.0", "id": i,
                "result": {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "echo", "version": "0.1"}}}
    if m == "tools/list":
        return {"jsonrpc": "2.0", "id": i, "result": {"tools": TOOLS}}
    if m == "tools/call":
        args = (req.get("params") or {}).get("arguments") or {}
        return {"jsonrpc": "2.0", "id": i,
                "result": {"content": [{"type": "text",
                                        "text": "echo: " + str(args.get("text", ""))}]}}
    return {"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": f"未知方法 {m}"}}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    sys.stdout.write(json.dumps(handle(req), ensure_ascii=False) + "\\n")
    sys.stdout.flush()
'''
