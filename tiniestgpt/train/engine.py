"""训练引擎：把"数据 + 模型 + 优化器 + 精度 + 分布式"组织成一个可观测的循环。

刻意保留的**工程细节**（每一项都是真实训练里会踩的坑）：
  * bf16  autocast + fp32 主权重（参数始终 fp32，只在计算时降精度）；
  * fp16 时才需要 GradScaler（bf16 的动态范围足够，不需要）；
  * 梯度裁剪前必须 ``unscale_``，否则裁剪阈值被 scale 放大；
  * **NaN/Inf 守卫**：出现坏梯度时跳过这一步而不污染模型；
  * 吞吐统计用"每步 token 数 / 耗时"，再乘以 FLOPs/token 得到 MFU。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..common.logging import get_logger
from ..common.profiler import AvgMeter, MFUTracker, Timer
from ..common.seed import init_device, set_seed
from ..data.dataloader import DataLoaderConfig, ShardedTokenDataset
from ..model.config import ModelConfig
from ..model.factory import build_model
from .checkpoint import find_latest, load_checkpoint, save_checkpoint
from .distributed import init_distributed, reduce_metrics, wrap_model
from .losses import language_modeling_loss, mtp_loss, perplexity
from .lr_sched import build_scheduler
from .optim import build_optimizer

log = get_logger("tiniestgpt.train")

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, cfg, model: Optional[nn.Module] = None) -> None:
        self.cfg = cfg
        self.rank, self.world_size, self.local_rank = init_distributed()
        set_seed(cfg.seed + self.rank)
        self.device = init_device(prefer_gpu=True)

        if isinstance(cfg.model, dict):
            cfg.model = ModelConfig(**cfg.model)
        cfg.model.attn_backend = cfg.attn_backend
        cfg.model.gradient_checkpointing = cfg.gradient_checkpointing

        self.model = model if model is not None else build_model(cfg.model)
        self.model.to(self.device)
        if cfg.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile enabled")
            except Exception as exc:
                log.warning("torch.compile 失败，回退: %s", exc)
        self.model = wrap_model(self.model, cfg.distributed, cfg)
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model

        self.opt = build_optimizer(cfg, self.raw_model)
        self.sched = build_scheduler(cfg, self.opt)

        # ---------------- 数据 ----------------
        dl_cfg = DataLoaderConfig(
            shard_dir=cfg.data_dir, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
            shuffle=True, seed=cfg.seed, doc_mask=cfg.doc_mask, num_prefetch=cfg.num_prefetch,
            device=self.device,
        )
        self.ds = ShardedTokenDataset(cfg.data_dir, dl_cfg)
        self.loader = self.ds.iter_batches()
        eval_ds = ShardedTokenDataset(cfg.data_dir, dl_cfg)
        self.eval_iter = eval_ds.iter_batches()

        # ---------------- 精度 ----------------
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.dtype, None)
        use_amp = self.amp_dtype is not None and self.device.type == "cuda"
        self.amp_enabled = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == "fp16" and use_amp))

        # ---------------- 训练稳定化：MuonClip / QK-Clip ----------------
        # 前向时顺带观测逐 head 的 max attention logit，每 N 步把越界的 Q/K 拉回安全区。
        self.qk_clip = None
        if getattr(cfg, "qk_clip_tau", 0) > 0:
            from .qk_clip import QKClipGuard

            self.qk_clip = QKClipGuard(
                self.raw_model, tau=cfg.qk_clip_tau,
                alpha=getattr(cfg, "qk_clip_alpha", 0.5),
                every=getattr(cfg, "qk_clip_every", 50),
            ).install()
            log.info("QK-Clip enabled: tau=%.1f every=%d", cfg.qk_clip_tau, cfg.qk_clip_every)

        # ---------------- 状态 ----------------
        self.step = 0
        self.best_eval = float("inf")
        self.loss_meter = AvgMeter(window=50)
        self.step_timer = Timer(sync_cuda=self.device.type == "cuda")
        self.mfu = MFUTracker(self.raw_model.flops_per_token(cfg.seq_len))
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if cfg.resume:
            self._resume(cfg.resume)

        log.info("\n%s", self.raw_model.summary())

    # ------------------------------------------------------------------ #
    def _resume(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            found = find_latest(p)
            if found is None:
                log.warning("目录 %s 下没有 checkpoint", p)
                return
            p = found
        ckpt = load_checkpoint(p, self.raw_model, self.opt, self.sched, map_location="cpu")
        self.step = int(ckpt.get("step", 0))
        log.info("resumed from %s (step=%d)", p, self.step)

    # ------------------------------------------------------------------ #
    def _compute_loss(self, batch: Dict[str, torch.Tensor]):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        attn_mask = batch.get("attn_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.amp_enabled):
            logits = self.model(input_ids, attn_mask=attn_mask)
        loss_d = language_modeling_loss(
            logits, labels,
            label_smoothing=self.cfg.label_smoothing,
            z_loss_weight=self.cfg.z_loss_weight,
            aux_loss=self.raw_model.last_aux_loss,
            aux_weight=self.cfg.moe_aux_weight,
        )
        # MTP 辅助损失：只加监督信号，不改变推理时的主 logits
        if getattr(self.cfg, "mtp_loss_weight", 0) > 0 and self.raw_model.last_mtp_logits:
            loss_d["mtp"] = mtp_loss(self.raw_model.last_mtp_logits, labels,
                                     weight=self.cfg.mtp_loss_weight)["mtp"]
            loss_d["loss"] = loss_d["loss"] + loss_d["mtp"]
        else:
            loss_d["mtp"] = torch.zeros((), device=logits.device)
        return loss_d

    # ------------------------------------------------------------------ #
    def train_step(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        loss_d = self._compute_loss(batch)
        (loss_d["loss"] / cfg.grad_accum_steps).backward()

        stats = {"ce": float(loss_d["ce"]), "z": float(loss_d["z"]), "aux": float(loss_d["aux"]),
                 "mtp": float(loss_d["mtp"])}
        return stats

    def _optimizer_step(self) -> float:
        cfg = self.cfg
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)
        grad_norm = float(grad_norm)
        # NaN / Inf 守卫：坏梯度直接跳过，避免污染整个模型
        if grad_norm != grad_norm or grad_norm in (float("inf"), float("-inf")):
            log.warning("step %d: 非法梯度 (norm=%s)，跳过更新", self.step, grad_norm)
            self.opt.zero_grad(set_to_none=True)
            return grad_norm
        if self.scaler.is_enabled():
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        return grad_norm

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.raw_model.eval()
        losses = []
        for _ in range(self.cfg.eval_batches):
            batch = next(self.eval_iter)
            d = self._compute_loss(batch)
            losses.append(float(d["ce"]))
        self.raw_model.train()
        return {"eval_ce": sum(losses) / max(len(losses), 1),
                "eval_ppl": perplexity(sum(losses) / max(len(losses), 1))}

    # ------------------------------------------------------------------ #
    def train(self, max_steps: Optional[int] = None) -> None:
        cfg = self.cfg
        max_steps = max_steps or cfg.max_steps
        self.raw_model.train()
        log.info("start training: %d steps, %d tokens/step, device=%s",
                 max_steps, cfg.tokens_per_step(), self.device)

        while self.step < max_steps:
            self.step_timer.start()
            self.opt.zero_grad(set_to_none=True)
            acc_stats = {"ce": 0.0, "z": 0.0, "aux": 0.0}
            for _ in range(cfg.grad_accum_steps):
                batch = next(self.loader)
                s = self.train_step(batch)
                for k in acc_stats:
                    acc_stats[k] += s[k] / cfg.grad_accum_steps
            grad_norm = self._optimizer_step()
            if self.qk_clip is not None:
                self.qk_clip.maybe_clip(self.step)
            dt = self.step_timer.stop()
            self.step += 1

            n_tokens = cfg.tokens_per_step() * self.world_size
            self.mfu.update(n_tokens, dt)
            self.loss_meter.update(acc_stats["ce"])

            if self.step % cfg.log_every == 0:
                metrics = reduce_metrics({
                    "ce": acc_stats["ce"], "grad_norm": grad_norm,
                }, device=self.device)
                log.info(
                    "step %6d/%d | loss %.4f (ppl %.1f) | aux %.4f | gn %.2f | lr %.2e | %s | %.1f GB",
                    self.step, max_steps, self.loss_meter.avg, perplexity(self.loss_meter.avg),
                    acc_stats["aux"], metrics["grad_norm"], max(self.sched.get_last_lr()),
                    self.mfu.report(),
                    torch.cuda.max_memory_allocated() / 1024 ** 3 if self.device.type == "cuda" else 0.0,
                )

            if cfg.eval_every and self.step % cfg.eval_every == 0:
                ev = self.evaluate()
                log.info("  [eval] step %d | ce %.4f | ppl %.2f", self.step, ev["eval_ce"], ev["eval_ppl"])
                if ev["eval_ce"] < self.best_eval:
                    self.best_eval = ev["eval_ce"]

            if cfg.save_every and self.step % cfg.save_every == 0:
                self.save()

        self.save(tag="last")
        log.info("training done. best eval ce = %.4f", self.best_eval)

    # ------------------------------------------------------------------ #
    def save(self, tag: Optional[str] = None) -> Path:
        name = tag if tag else f"ckpt_{self.step:07d}"
        path = self.out_dir / f"{name}.pt"
        save_checkpoint(
            path, self.raw_model, self.opt, self.sched, step=self.step,
            # "config" 必须是 ModelConfig（推理侧 load_model 依赖它）
            cfg=vars(self.raw_model.cfg), async_save=False,
            extra={"best_eval": self.best_eval, "model_summary": self.raw_model.summary(),
                   "train_config": vars(self.cfg)},
        )
        log.info("saved checkpoint -> %s", path)
        return path
