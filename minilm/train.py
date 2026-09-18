"""Training loop, evaluation, and checkpoint recovery.

Covers:
- Session 2: Training-loop anatomy (gradient accumulation, clipping, AMP, checkpointing).
- Session 4: Train a small GPT (training execution, validation monitoring).
"""

import os
import time
from contextlib import nullcontext
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from .optim import get_lr_cosine_schedule


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    device_type: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> float:
    """Execute forward pass, loss calculation, backprop, and optional optimizer step."""
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=dtype)
        if device_type in ("cuda", "cpu") and dtype != torch.float32
        else nullcontext()
    )

    with ctx:
        logits, loss, _ = model(x, targets=y)
        loss = loss / grad_accum_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item() * grad_accum_steps


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Clip gradients, step optimizer, and reset gradients to zero."""
    if scaler is not None:
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return norm.item() if isinstance(norm, torch.Tensor) else float(norm)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset,
    eval_iters: int = 20,
    batch_size: int = 4,
    block_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Estimate validation loss over a fixed number of iterations."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size, block_size, device=device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    step: int,
    val_loss: float,
) -> None:
    """Save model checkpoint with optimizer state and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load model weights and optimizer state from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
