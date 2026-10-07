"""轻量日志：rank 感知、rich 可选、避免多进程重复刷屏。"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

__all__ = ["get_logger", "setup_logging"]

_CONFIGURED = False
_LEVEL = os.environ.get("TINIESTGPT_LOG_LEVEL", "INFO").upper()

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: Optional[str] = None, force: bool = False) -> None:
    """初始化 root logger。多进程训练时只有 rank0 输出到 stderr。"""
    global _CONFIGURED, _LEVEL
    if _CONFIGURED and not force:
        return
    _LEVEL = (level or _LEVEL).upper()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler  # type: ignore

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(name)s | %(message)s", datefmt=_DATEFMT))
    except Exception:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    root.addHandler(handler)
    root.setLevel(_LEVEL)
    # 让第三方库安静一点
    for noisy in ("urllib3", "asyncio", "matplotlib", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取 logger；非主进程自动降级为 WARNING，避免 DDP 下日志爆炸。"""
    setup_logging()
    log = logging.getLogger(name)
    try:
        from .seed import is_main_process

        if not is_main_process():
            log.setLevel(max(log.level or 0, logging.WARNING))
    except Exception:  # pragma: no cover
        pass
    return log
