"""内置工具：一个"够用且安全"的工具箱。

安全边界（很重要）：
  * ``python_repl`` 默认**关闭**，且走**子进程 + 超时**，
    进程退出后状态不保留（不污染 Agent 进程）；
  * ``read_file`` / ``write_file`` 被限制在 ``root_dir`` 之内（防路径穿越）；
  * ``calculator`` 用 AST 白名单求值，而不是 ``eval``；
  * ``web_search`` 是**离线占位实现**，需要联网时替换成真实 API 即可。
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import ToolRegistry, tool

__all__ = ["build_default_registry", "calculator", "think", "now", "read_file", "write_file",
           "list_dir", "python_repl", "python_sandbox", "todo_write", "rag_search", "web_search"]


def _default_root() -> Path:
    return Path(os.environ.get("TINIESTGPT_FS_ROOT", ".")).resolve()


# --------------------------------------------------------------------------- #
@tool(name="calculator", description="计算数学表达式，支持 + - * / ** 与常见数学函数")
def calculator(expression: str) -> str:
    """求值一个数学表达式，例如 "2**10 + 3*(4+5)"。"""
    # 注意：Python 3.12+ 移除了 ast.Num，数字统一由 ast.Constant 表示
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                     ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
                     ast.Call, ast.Name, ast.Load, ast.Tuple)
    safe_names = {k: getattr(math, k) for k in
                  ("sqrt", "sin", "cos", "tan", "log", "log2", "log10", "exp", "floor",
                   "ceil", "pi", "e", "pow", "fabs", "factorial")}
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"表达式语法错误: {exc}"
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return f"不允许的语法: {type(node).__name__}"
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            return "只允许调用白名单函数"
    try:
        val = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, safe_names)  # noqa: S307
    except Exception as exc:
        return f"计算失败: {exc}"
    return str(val)


@tool(name="think", description="把中间推理过程记录下来（不产生任何副作用）")
def think(text: str) -> str:
    """记录一段思考，用于让推理链可追溯。"""
    return f"已记录思考（{len(text)} 字）"


@tool(name="now", description="获取当前时间（ISO 8601 格式）")
def now() -> str:
    """返回当前时间。"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# 文件系统（限制在 root 内）
# --------------------------------------------------------------------------- #
def _safe_path(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root)):
        raise PermissionError(f"路径越界: {rel}")
    return p


@tool(name="read_file", description="读取文件内容（限制在工作目录内）")
def read_file(path: str, max_chars: int = 4000) -> str:
    """读取文本文件的前 max_chars 个字符。"""
    p = _safe_path(_default_root(), path)
    if not p.exists():
        return f"文件不存在: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


@tool(name="write_file", description="把内容写入文件（限制在工作目录内）")
def write_file(path: str, content: str) -> str:
    """写入文件，返回写入字节数。"""
    p = _safe_path(_default_root(), path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {len(content)} 字符 -> {p}"


@tool(name="list_dir", description="列出目录下的文件（限制在工作目录内）")
def list_dir(path: str = ".") -> str:
    """列出目录内容。"""
    p = _safe_path(_default_root(), path)
    if not p.is_dir():
        return f"不是目录: {path}"
    items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(items[:200]) if items else "(空目录)"


# --------------------------------------------------------------------------- #
@tool(name="python_repl", description="在独立子进程中执行 Python 代码（有超时；默认禁用）",
      dangerous=True)
def python_repl(code: str, timeout: float = 5.0, enabled: bool = False) -> str:
    """执行 Python 代码并返回 stdout。

    走 :mod:`tiniestgpt.agent.sandbox`：**受限子进程** + 墙钟超时 +
    CPU/内存限额 + 禁用网络 + 输出截断。进程退出后不保留状态。
    """
    if not enabled:
        return "python_repl 已禁用：创建工具时设置 allow_python=True 以启用"
    from .sandbox import SandboxConfig, run_python

    cfg = SandboxConfig(timeout=float(timeout), cpu_seconds=max(int(timeout), 1))
    return run_python(code, cfg).as_text()


@tool(name="python_sandbox", description="在持久化沙箱工作目录中执行 Python（文件会保留）",
      dangerous=True)
def python_sandbox(code: str, timeout: float = 5.0, enabled: bool = True) -> str:
    """在同一个沙箱工作目录里执行代码——**上一步写的文件下一步还在**。

    这是长程 Agent 的关键：没有持久化工作目录，
    Agent 就无法"先写脚本、再跑脚本、再改脚本"。
    """
    if not enabled:
        return "python_sandbox 已禁用"
    from .sandbox import Sandbox, SandboxConfig

    global _SANDBOX
    if _SANDBOX is None:
        _SANDBOX = Sandbox(SandboxConfig(timeout=float(timeout)))
    return _SANDBOX.run(code).as_text()


_SANDBOX = None


# --------------------------------------------------------------------------- #
@tool(name="todo_write", description="记录或更新待办列表")
def todo_write(items: str) -> str:
    """把待办事项写下来（用分号或换行分隔）。"""
    parts = [x.strip() for x in items.replace("\n", ";").split(";") if x.strip()]
    return "待办已更新：\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))


@tool(name="web_search", description="搜索网页（离线占位实现，可替换为真实 API）")
def web_search(query: str, k: int = 3) -> str:
    """占位实现：返回说明性文本。接入真实搜索只需替换本函数。"""
    return (f"[web_search 占位] 查询：{query}\n"
            "本实现刻意不联网以保证离线可运行；"
            "替换本函数即可接入 Bing/Serper/Tavily 等真实搜索 API。")


# --------------------------------------------------------------------------- #
def build_default_registry(allow_python: bool = False, allow_write: bool = True,
                           vector_memory=None, root: Optional[str] = None) -> ToolRegistry:
    """构建一个默认工具集（可按安全等级裁剪）。

    :param allow_python: 是否开放 ``python_repl``（子进程执行）
    :param allow_write:  是否开放写文件
    :param vector_memory: 传入后启用 ``rag_search``（本地向量检索）
    """
    if root:
        os.environ["TINIESTGPT_FS_ROOT"] = str(Path(root).resolve())
    reg = ToolRegistry()
    reg.register_fn(calculator)
    reg.register_fn(think)
    reg.register_fn(now)
    reg.register_fn(read_file)
    reg.register_fn(list_dir)
    if allow_write:
        reg.register_fn(write_file)
    if allow_python:
        reg.register_fn(python_repl)
    reg.register_fn(todo_write)
    reg.register_fn(web_search)

    if vector_memory is not None:
        def rag(query: str, k: int = 3) -> str:
            hits = vector_memory.search(query, k)
            if not hits:
                return "(没有检索到相关记忆)"
            return "\n---\n".join(f"(相似度 {h.score:.2f}) {h.text}" for h in hits)

        rag.__name__ = "rag_search"
        rag.__doc__ = "在本地向量记忆中检索与问题相关的历史信息"
        reg.register_fn(rag)

    return reg


def rag_search(query: str, k: int = 3) -> str:      # 便于外部直接引用
    return "(rag_search 需要传入 vector_memory)"
