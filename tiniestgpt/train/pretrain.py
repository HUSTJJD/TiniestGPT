"""预训练入口：``python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config import apply_overrides, from_dict, load_config, save_config
from ..common.logging import get_logger
from .config import TrainConfig
from .engine import Trainer

log = get_logger("tiniestgpt.pretrain")

__all__ = ["main"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("tiniestgpt pretrain")
    p.add_argument("--config", type=str, default=None, help="YAML/JSON 配置")
    p.add_argument("--preset", type=str, default=None, help="模型预设名（覆盖 config.model）")
    p.add_argument("--set", nargs="*", default=[], help="点号覆盖，如 train.lr=1e-3")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    if args.config:
        cfg = load_config(TrainConfig, args.config, args.set)
    else:
        cfg = from_dict(TrainConfig, apply_overrides({}, args.set))

    if args.preset:
        from ..model.config import PRESETS

        cfg.model = PRESETS[args.preset]
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.resume:
        cfg.resume = args.resume

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    trainer = Trainer(cfg)
    trainer.train()
