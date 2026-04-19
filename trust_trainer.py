"""
Trust-weighted training loop for GPT-2 + Engram.

Key idea: every training example already carries a TrustSignal.  We use
trust_score ∈ [0,1] as a per-example loss weight so that:
  - high-trust examples (score ≈ 1) drive full gradient updates
  - low-trust / contradictory examples (score ≈ 0) contribute near-zero
    gradient — the neural equivalent of the symbolic quarantine in
    engram_trust.py

Engram parameters get a 5× learning rate (paper §4.1) with no weight
decay; backbone parameters use standard AdamW settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from engram_backbone import EngramBackbone
from engram_trust import TrainingExample


@dataclass
class TrainingConfig:
    batch_size: int = 4
    seq_len: int = 256
    lr: float = 2e-5              # backbone learning rate (ignored when frozen)
    engram_lr_multiplier: float = 5.0    # paper: 5× LR for Engram params
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 1000
    warmup_steps: int = 50
    save_every: int = 250
    log_every: int = 25
    output_dir: str = "checkpoints"
    min_trust_weight: float = 0.0
    # When True (quantized backbone), only Engram params enter the optimizer
    freeze_backbone: bool = False


def build_optimizer(model: EngramBackbone, config: TrainingConfig) -> torch.optim.AdamW:
    """Optimizer with Engram at 5× LR (paper §4.1).

    When freeze_backbone=True (4-bit quantized model), only Engram parameters
    are included — backbone params have no gradients and must not be in any
    optimizer param group.
    """
    engram_params = list(model.engram.parameters())
    if config.freeze_backbone:
        return torch.optim.AdamW(
            [{"params": engram_params, "lr": config.lr * config.engram_lr_multiplier, "weight_decay": 0.0}]
        )

    engram_ids = {id(p) for p in engram_params}
    backbone_params = [p for p in model.parameters() if id(p) not in engram_ids and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr, "weight_decay": config.weight_decay},
            {"params": engram_params, "lr": config.lr * config.engram_lr_multiplier, "weight_decay": 0.0},
        ]
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to 10% of peak LR."""
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        import math
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def tokenize_examples(
    examples: Sequence[TrainingExample],
    tokenizer,
    config: TrainingConfig,
) -> List[tuple[torch.Tensor, float]]:
    """Return list of (token_id_tensor, trust_weight) pairs, skipping quarantined."""
    pairs: List[tuple[torch.Tensor, float]] = []
    for ex in examples:
        trust = ex.signals.score()
        if trust <= config.min_trust_weight:
            continue
        # Concatenate text + payload as the language-model target sequence
        full_text = ex.text.strip() + "\n" + ex.payload.strip()
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=config.seq_len,
            return_tensors="pt",
        )
        pairs.append((enc["input_ids"][0], trust))
    return pairs


def iter_batches(
    pairs: Sequence[tuple[torch.Tensor, float]],
    batch_size: int,
    shuffle: bool = True,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (padded_input_ids [B, T], trust_scores [B]) batches."""
    indices = list(range(len(pairs)))
    if shuffle:
        import random
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        chunk = [pairs[indices[i]] for i in indices[start : start + batch_size]]
        if not chunk:
            continue
        ids_list, trusts = zip(*chunk)
        max_len = max(t.size(0) for t in ids_list)
        padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            padded[i, : ids.size(0)] = ids
        yield padded, torch.tensor(trusts, dtype=torch.float32)


# ------------------------------------------------------------------
# Training step
# ------------------------------------------------------------------

def train_step(
    model: EngramBackbone,
    input_ids: torch.Tensor,
    trust_scores: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    config: TrainingConfig,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    trust_scores = trust_scores.to(device)

    # Shift labels: predict next token at each position
    output = model(input_ids)
    logits = output.logits                          # [B, T, vocab]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Per-token cross-entropy, then mean over tokens per example
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)                  # [B, T-1]

    per_example = per_token.mean(dim=-1)            # [B]

    # Trust-weighted batch loss: high-trust examples drive stronger updates
    loss = (per_example * trust_scores).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "mean_trust": float(trust_scores.mean()),
    }


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(
    model: EngramBackbone,
    examples: Sequence[TrainingExample],
    tokenizer,
    config: TrainingConfig,
) -> List[Dict[str, object]]:
    """
    Train the model for config.max_steps steps.

    Returns a list of per-log-step metric dicts for later analysis.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Tokenizing {len(examples)} examples …")
    pairs = tokenize_examples(examples, tokenizer, config)
    print(
        f"  {len(pairs)} examples after trust filtering "
        f"(min_trust={config.min_trust_weight})"
    )
    if not pairs:
        raise ValueError("No training examples after trust filtering — lower min_trust_weight.")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    model.train()
    log: List[Dict[str, object]] = []
    step = 0

    while step < config.max_steps:
        for input_ids, trust_scores in iter_batches(pairs, config.batch_size):
            if step >= config.max_steps:
                break

            metrics = train_step(model, input_ids, trust_scores, optimizer, scheduler, config)
            step += 1

            if step % config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                entry: Dict[str, object] = {"step": step, "lr": round(lr, 8), **{
                    k: round(v, 4) for k, v in metrics.items()
                }}
                log.append(entry)
                print(
                    f"step {step:5d}  loss {metrics['loss']:.4f}  "
                    f"grad {metrics['grad_norm']:.3f}  "
                    f"trust {metrics['mean_trust']:.3f}  lr {lr:.2e}"
                )

            if step % config.save_every == 0:
                ckpt = output_dir / f"checkpoint_step{step:05d}.pt"
                # Save only Engram weights — backbone reloads from HuggingFace at eval time
                engram_state = {f"engram.{k}": v for k, v in model.engram.state_dict().items()}
                torch.save({"step": step, "state_dict": engram_state}, ckpt)
                print(f"  -> saved {ckpt}")

    return log
