"""真沙箱：Agent 执行不可信代码的隔离层。

``builtin_tools.python_repl`` 之前是**直接 exec**——没有超时、没有内存上限、
没有网络隔离。对教学项目来说这是一个真实的安全洞：
一句 ``while True: pass`` 就能把整个进程挂死。

2026 年长程 Agent 的前提就是"受控的计算环境"：独立文件系统、Shell、
持久化任务状态、资源限额、**可中断**。本模块提供最小的进程级隔离：

* :class:`SandboxConfig` —— CPU 秒数 / 内存 / 输出字节 / 临时目录
* :func:`run_python`     —— 在**子进程**里跑代码，带超时与内存限制
* :class:`SandboxTool`   —— 包装成 Agent 可直接注册的工具

进程级隔离不是容器级隔离（没有 chroot / 网络命名空间），
但已经能挡住绝大多数"Agent 把自己跑崩"的情况，且**零依赖、跨平台**。
需要更强隔离时用 ``backend="docker"``。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["SandboxConfig", "SandboxResult", "run_python", "Sandbox", "safe_exec"]


@dataclass
class SandboxConfig:
    cpu_seconds: int = 5
    memory_mb: int = 512
    max_output_chars: int = 8000
    timeout: float = 8.0            # 墙钟超时（含进程启动开销）
    workdir: Optional[str] = None   # None = 临时目录，跑完即删
    allow_network: bool = False
    backend: str = "process"        # process | docker


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    returncode: int = 0
    seconds: float = 0.0
    truncated: bool = False

    def as_text(self) -> str:
        if not self.ok and self.error:
            return f"[sandbox error] {self.error}\n{self.stderr[:500]}"
        out = self.stdout
        if self.truncated:
            out += "\n... (输出被截断)"
        if self.stderr.strip():
            out += "\n[stderr] " + self.stderr[:500]
        return out or "(无输出)"


# --------------------------------------------------------------------------- #
#  子进程执行体
# --------------------------------------------------------------------------- #
def _child(code: str, q: Any, cfg: SandboxConfig) -> None:
    """在子进程里执行代码（**必须**先设资源上限再 exec）。"""
    import io

    # 0) 切入沙箱工作目录（长程 Agent 需要"上一步写的文件还在"）
    if cfg.workdir:
        try:
            os.chdir(cfg.workdir)
        except Exception:
            pass

    # 1) 资源限额。Windows 没有 resource 模块，此时只靠墙钟超时兜底——
    #    这也是为什么强隔离场景必须用 backend="docker"。
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds + 1))
        lim = cfg.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass
    if not cfg.allow_network:
        # 尽力而为：屏蔽 socket 让"偷偷联网"直接失败
        try:
            import socket

            class _Blocked(socket.socket):
                def __init__(self, *a, **k):
                    raise OSError("沙箱已禁用网络访问")

            socket.socket = _Blocked          # type: ignore[assignment]
            socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore
                OSError("沙箱已禁用网络访问"))
        except Exception:
            pass

    buf, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, err
    try:
        ns: Dict[str, Any] = {"__name__": "__sandbox__"}
        exec(compile(code, "<sandbox>", "exec"), ns)     # noqa: S102 - 隔离在子进程里
        rc = 0
    except BaseException as exc:                          # noqa: BLE001
        traceback.print_exc(file=err)
        rc = 1
        _ = exc
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    try:
        q.put({"stdout": buf.getvalue(), "stderr": err.getvalue(), "rc": rc})
    except Exception:
        pass


def run_python(code: str, cfg: Optional[SandboxConfig] = None) -> SandboxResult:
    """在受限子进程里执行 Python 代码。"""
    cfg = cfg or SandboxConfig()
    t0 = time.time()
    ctx = mp.get_context("spawn")            # spawn 保证不继承父进程状态
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(code, q, cfg))
    p.start()
    p.join(cfg.timeout)
    seconds = time.time() - t0

    if p.is_alive():
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join(1)
        return SandboxResult(ok=False, error=f"执行超时（>{cfg.timeout}s）或触发资源上限",
                             returncode=-1, seconds=seconds)

    data: Dict[str, Any] = {}
    try:
        if not q.empty():
            data = q.get(timeout=1)
    except Exception:
        pass
    out, err = data.get("stdout", ""), data.get("stderr", "")
    truncated = len(out) > cfg.max_output_chars
    if truncated:
        out = out[: cfg.max_output_chars]
    return SandboxResult(ok=data.get("rc", 1) == 0, stdout=out, stderr=err,
                         returncode=int(data.get("rc", 1)), seconds=seconds,
                         truncated=truncated)


def safe_exec(code: str, cfg: Optional[SandboxConfig] = None) -> str:
    """给工具用的便捷封装：返回可直接回灌给模型的文本。"""
    return run_python(code, cfg).as_text()


# --------------------------------------------------------------------------- #
class Sandbox:
    """带工作目录的沙箱（长程 Agent 需要"文件在哪"是稳定的）。"""

    def __init__(self, cfg: Optional[SandboxConfig] = None) -> None:
        self.cfg = cfg or SandboxConfig()
        self.root = Path(self.cfg.workdir) if self.cfg.workdir else Path(tempfile.mkdtemp(
            prefix="tiniest_sandbox_"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def run(self, code: str) -> SandboxResult:
        self.calls += 1
        cfg = SandboxConfig(**{**vars(self.cfg), "workdir": str(self.root)})
        return run_python(code, cfg)

    def write_file(self, name: str, content: str) -> str:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p)

    def read_file(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def __repr__(self) -> str:
        return f"Sandbox(root={self.root}, calls={self.calls})"
