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
    # TODO: run the model under the appropriate autocast context.
    # TODO: divide loss by grad_accum_steps before backward.
    # TODO: support both an AMP scaler and the ordinary backward path.
    raise NotImplementedError


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Clip gradients, update parameters, and clear gradient buffers."""
    # TODO: unscale first when using AMP, then clip before optimizer.step().
    # TODO: clear gradients with set_to_none=True after the update.
    raise NotImplementedError


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    step: int,
    val_loss: float,
) -> None:
    """Save model, optimizer, configuration, and progress metadata."""
    raise NotImplementedError


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Restore model state and, when supplied, optimizer state."""
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
