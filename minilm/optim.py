"""Optimizer and Learning Rate Schedule configuration.

Covers:
- Session 2: Training-loop anatomy (AdamW weight decay grouping, learning rate schedules).
"""

import math
from typing import List, Dict, Any
import torch
import torch.nn as nn


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.Optimizer:
    """Create optimizer with decoupled weight decay.

    Rules:
    - All 2D parameters (weights in Linear and Embedding layers) will be weight-decayed.
    - All 1D parameters (biases and LayerNorm gain/bias) will NOT be weight-decayed.
    """
    decay_params: List[nn.Parameter] = []
    nodecay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups: List[Dict[str, Any]] = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer with fused kernel if available on CUDA
    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    return optimizer


def get_lr_cosine_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute learning rate using linear warmup followed by cosine decay."""
    # 1) Linear warmup for warmup_steps steps
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    # 2) If step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) In between, use cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1.0 -> 0.0
    return min_lr + coeff * (max_lr - min_lr)
