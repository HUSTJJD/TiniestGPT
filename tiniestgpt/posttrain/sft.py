"""监督微调（SFT）。

两个容易做错、但决定效果的细节：
  * **prompt masking**：只让 response 参与 loss。若把 prompt 也算进去，
    模型会去学"复述问题"，且在短回答上梯度被稀释。
  * **packing + 不跨样本注意力**：多条样本拼成定长序列能提高吞吐，
    但必须配合块对角掩码（否则样本之间会互相"泄题"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["SFTConfig", "SFTExample", "build_sft_batch", "sft_loss", "SFTTrainer"]


@dataclass
class SFTExample:
    prompt: str
    response: str
    weight: float = 1.0


@dataclass
class SFTConfig:
    max_len: int = 512
    batch_size: int = 4
    lr: float = 1e-5
    epochs: int = 2
    warmup_ratio: float = 0.03
    grad_accum_steps: int = 4
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    mask_prompt: bool = True
    packing: bool = True
    log_every: int = 10
    out_dir: str = "out/sft"


def build_sft_batch(
    examples: Sequence[SFTExample],
    tokenizer,
    cfg: SFTConfig,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """把若干样本拼成一条定长序列（packing），并返回 loss 掩码与块对角掩码。"""
    B = len(examples)
    T = cfg.max_len
    ids = torch.full((B, T), tokenizer.pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    doc_ids = torch.full((B, T), -1, dtype=torch.long)

    for b, ex in enumerate(examples):
        p_ids = tokenizer.encode(ex.prompt, add_bos=True)
        r_ids = tokenizer.encode(ex.response, add_eos=True)
        seq = (p_ids + r_ids)[:T]
        n = len(seq)
        ids[b, :n] = torch.tensor(seq, dtype=torch.long)
        doc_ids[b, :n] = b
        if cfg.mask_prompt:
            start = min(len(p_ids), n)
        else:
            start = 0
        labels[b, start:n] = torch.tensor(seq[start:n], dtype=torch.long)

    input_ids = ids[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    doc_ids = doc_ids[:, 1:].contiguous()

    batch: Dict[str, torch.Tensor] = {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
    }
    if cfg.packing and B > 1:
        same = doc_ids[:, :, None] == doc_ids[:, None, :]
        causal = torch.tril(torch.ones(T - 1, T - 1, dtype=torch.bool))
        eye = torch.eye(T - 1, dtype=torch.bool).unsqueeze(0)
        batch["attn_mask"] = ((same & causal.unsqueeze(0)) | eye).unsqueeze(1).to(device)
    return batch


def sft_loss(logits: torch.Tensor, labels: torch.Tensor,
             label_smoothing: float = 0.0) -> torch.Tensor:
    B, T, V = logits.shape
    return F.cross_entropy(logits.float().reshape(-1, V), labels.reshape(-1),
                           ignore_index=-100, label_smoothing=label_smoothing)


class SFTTrainer:
    """极简 SFT 训练器：不追求分布式，追求"每一行都看得懂"。"""

    def __init__(self, model: torch.nn.Module, cfg: SFTConfig,
                 device: torch.device | str = "cpu") -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0,
                                     betas=(0.9, 0.95))
        self.global_step = 0

    def fit(self, examples: Sequence[SFTExample], tokenizer, epochs: Optional[int] = None) -> None:
        cfg = self.cfg
        epochs = epochs or cfg.epochs
        n_batches = max(len(examples) // cfg.batch_size, 1)
        total_steps = n_batches * epochs // max(cfg.grad_accum_steps, 1)
        warmup = max(int(total_steps * cfg.warmup_ratio), 1)

        def lr_at(step: int) -> float:
            if step < warmup:
                return cfg.lr * (step + 1) / warmup
            t = (step - warmup) / max(total_steps - warmup, 1)
            return cfg.lr * 0.1 + (cfg.lr * 0.9) * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(t, 1.0))).item())

        self.model.train()
        self.model.to(self.device)
        for ep in range(epochs):
            order = torch.randperm(len(examples)).tolist()
            for bi in range(n_batches):
                sel = [examples[i] for i in order[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]]
                batch = build_sft_batch(sel, tokenizer, cfg, self.device)
                logits = self.model(batch["input_ids"], attn_mask=batch.get("attn_mask"))
                loss = sft_loss(logits, batch["labels"], cfg.label_smoothing)
                (loss / cfg.grad_accum_steps).backward()
                if (bi + 1) % cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    lr = lr_at(self.global_step)
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.global_step % cfg.log_every == 0:
                        print(f"[sft] epoch {ep} step {self.global_step} loss {loss.item():.4f} lr {lr:.2e}")
