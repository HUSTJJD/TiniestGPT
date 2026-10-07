"""通用配置系统。

提供 dataclass ⇄ dict / JSON / YAML 的双向转换，支持：
  * 嵌套 dataclass、Optional、Union、List / Dict / Tuple、Enum
  * YAML/JSON 文件加载（无 pyyaml 时自动退化为 JSON）
  * 命令行点号覆盖：``--set model.n_layers=8 train.lr=3e-4``

之所以自己写而不用 pydantic / hydra：让"配置如何变成对象"这件事完全可见，
同时也少一个黑盒依赖。
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union,
                    get_args, get_origin, get_type_hints)

T = TypeVar("T")

__all__ = ["load_config", "save_config", "from_dict", "to_dict", "apply_overrides", "dump_defaults"]


# --------------------------------------------------------------------------- #
# 类型工具
# --------------------------------------------------------------------------- #
def _unwrap_optional(tp: Any) -> Any:
    """把 Optional[X] / Union[X, None] 归一为 X。"""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0]
    return tp


def _coerce(tp: Any, value: Any) -> Any:
    """把原始（通常来自 YAML/JSON）的值强制转换成目标类型 tp。"""
    if tp is Any or tp is None:
        return value
    if value is None:
        return None

    tp = _unwrap_optional(tp)
    origin = get_origin(tp)

    # 嵌套 dataclass
    if is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"期望 dict 以构造 {tp.__name__}，实际得到 {type(value).__name__}")
        return from_dict(tp, value)

    # 枚举：允许用名字或值
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return value if isinstance(value, tp) else tp(value)

    # 容器
    if origin in (list, List):
        item_tp = get_args(tp)[0] if get_args(tp) else Any
        return [_coerce(item_tp, v) for v in value]
    if origin in (tuple, Tuple):
        args = get_args(tp)
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        return tuple(_coerce(a, v) for a, v in zip(args, value))
    if origin in (dict, Dict):
        k_tp, v_tp = get_args(tp) if get_args(tp) else (Any, Any)
        return {_coerce(k_tp, k): _coerce(v_tp, v) for k, v in value.items()}

    # 标量
    if tp is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(tp, type) and not issubclass(tp, (str, bytes)):
        try:
            return tp(value)
        except Exception as exc:  # 给出可读的错误
            raise TypeError(f"无法把 {value!r} 转换为 {tp}: {exc}") from None
    return value


# --------------------------------------------------------------------------- #
# dataclass ⇄ dict
# --------------------------------------------------------------------------- #
def from_dict(cls: Type[T], data: Dict[str, Any], strict: bool = False) -> T:
    """从 dict 构造 dataclass；未知字段默认忽略（strict=True 时报错）。"""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} 不是 dataclass")
    hints = get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints.get(f.name, f.type), data[f.name])
        elif f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        elif strict:
            raise KeyError(f"缺少必需字段 {cls.__name__}.{f.name}")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown and strict:
        raise KeyError(f"{cls.__name__} 收到未知字段: {sorted(unknown)}")
    return cls(**kwargs)  # type: ignore[call-arg]


def to_dict(obj: Any) -> Any:
    """dataclass → 可 JSON 序列化的 dict（递归，枚举转为其值）。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        out: Dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# 文件 IO
# --------------------------------------------------------------------------- #
def _read_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("读取 YAML 需要 pyyaml：pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")


def load_config(cls: Type[T], path: str | Path, overrides: Optional[Iterable[str]] = None) -> T:
    """从 JSON/YAML 加载配置，并可选应用 ``a.b.c=value`` 形式的覆盖。"""
    data = _read_file(path)
    data = apply_overrides(data, overrides or [])
    return from_dict(cls, data)


def save_config(obj: Any, path: str | Path) -> None:
    """把配置写回 JSON/YAML。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(obj)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("写出 YAML 需要 pyyaml：pip install pyyaml") from exc
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_defaults(cls: Type[T]) -> Dict[str, Any]:
    """导出一个 dataclass 的全部默认值，便于生成配方模板。"""
    return to_dict(cls())


# --------------------------------------------------------------------------- #
# 命令行覆盖
# --------------------------------------------------------------------------- #
def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() in ("null", "none"):
        return None
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return json.loads(raw)  # 数字 / 数组 / 对象
    except Exception:
        return raw  # 字符串


def apply_overrides(data: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """把 ``--set a.b=1`` 这样的覆盖应用到配置字典上（原地修改并返回）。"""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"覆盖项格式应为 key=value，实际: {ov!r}")
        key, raw = ov.split("=", 1)
        value = _parse_value(raw)
        node: Dict[str, Any] = data
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return data
