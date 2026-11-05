"""Session 2 starter — backward pass, optimizer step, and checkpoints."""

from typing import Any, Optional

import torch
import torch.nn as nn


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
    """Run forward and backward, accumulating normalized gradients."""
    # TODO: start with the ordinary FP32 forward pass.
    # TODO: divide loss by grad_accum_steps before backward.
    # Optional Module 4 extension: support autocast and a gradient scaler.
    raise NotImplementedError


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Clip gradients, update parameters, and clear gradient buffers."""
    # TODO: clip before optimizer.step(). Optional AMP: unscale first.
    # TODO: clear gradients with set_to_none=True after the update.
    raise NotImplementedError


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    step: int,
    val_loss: float,
    extra_state: Optional[dict[str, Any]] = None,
) -> None:
    """Save model, optimizer, configuration, progress and optional recovery metadata.

    For baseline recovery, also save Python, NumPy and PyTorch RNG states.
    """
    raise NotImplementedError


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Restore model/optimizer and optionally the saved RNG states."""
    raise NotImplementedError


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: Any,
    eval_iters: int = 20,
    batch_size: int = 4,
    block_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Optional extension used by the longer baseline-training script."""
    raise NotImplementedError
