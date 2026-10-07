"""极简注册表：让"用字符串配置选实现"这件事标准化。

用法::

    OPTIM = Registry("optimizer")

    @OPTIM.register("adamw")
    def build_adamw(params, **kw): ...

    opt = OPTIM.build("adamw", params, lr=1e-3)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, TypeVar

T = TypeVar("T")

__all__ = ["Registry"]


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str, obj: Any = None) -> Callable[[Any], Any]:
        def _wrap(o: Any) -> Any:
            if key in self._items:
                raise KeyError(f"[{self.name}] 重复注册: {key}")
            self._items[key] = o
            return o

        return _wrap(obj) if obj is not None else _wrap

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"[{self.name}] 未注册: {key}；可用: {sorted(self._items)}")
        return self._items[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> Any:
        obj = self.get(key)
        return obj(*args, **kwargs)

    def keys(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry({self.name}, keys={self.keys()})"


# 全局注册表
MODEL = Registry("model")
OPTIMIZER = Registry("optimizer")
SCHEDULER = Registry("scheduler")
DATASET = Registry("dataset")
QUANTIZER = Registry("quantizer")
SAMPLER = Registry("sampler")
TOOL = Registry("tool")
PLANNER = Registry("planner")
BACKEND = Registry("llm_backend")
