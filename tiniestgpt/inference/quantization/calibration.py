"""校准数据采集：用 forward hook 抓每个 Linear 的输入激活。

GPTQ / AWQ / SmoothQuant 都需要"真实的激活分布"。
采集方式：注册 ``register_forward_pre_hook``，
跑若干条样本，把每个模块的输入张量攒下来（只保留需要的，避免爆内存）。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["CalibrationCollector", "collect_calibration_inputs"]


class CalibrationCollector:
    """收集指定模块（默认全部 nn.Linear）的输入激活。"""

    def __init__(self, model: nn.Module, target_types=(nn.Linear,),
                 max_samples_per_module: int = 2048,
                 name_filter: Optional[Sequence[str]] = None) -> None:
        self.model = model
        self.data: Dict[str, List[torch.Tensor]] = {}
        self.handles = []
        self.max_samples = max_samples_per_module
        self.name_filter = tuple(name_filter) if name_filter else None

        for name, mod in model.named_modules():
            if not isinstance(mod, target_types):
                continue
            if self.name_filter and not any(f in name for f in self.name_filter):
                continue
            self.data[name] = []
            self.handles.append(
                mod.register_forward_pre_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            flat = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            n = flat.shape[0]
            if n > self.max_samples:
                # 随机抽样子集，保持分布形状
                idx = torch.randint(0, n, (self.max_samples,))
                flat = flat[idx]
            self.data[name].append(flat)
        return hook

    @torch.no_grad()
    def collect(self, batches: Iterable, forward_fn=None, max_batches: int = 32) -> Dict[str, torch.Tensor]:
        """跑若干 batch，返回 {模块名: 拼接后的激活 [N, in]}。"""
        was_training = self.model.training
        self.model.eval()
        n = 0
        for batch in batches:
            if n >= max_batches:
                break
            if forward_fn is not None:
                forward_fn(batch)
            elif isinstance(batch, dict):
                self.model(batch["input_ids"])
            else:
                self.model(batch)
            n += 1
        self.remove()
        if was_training:
            self.model.train()
        return {k: torch.cat(v, dim=0) for k, v in self.data.items() if v}

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "CalibrationCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def collect_calibration_inputs(model: nn.Module, batches: Iterable, max_batches: int = 16,
                               max_samples: int = 2048, forward_fn=None) -> Dict[str, torch.Tensor]:
    """一步到位的便捷函数。"""
    with CalibrationCollector(model, max_samples_per_module=max_samples) as c:
        return c.collect(batches, forward_fn=forward_fn, max_batches=max_batches)
