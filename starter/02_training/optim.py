"""Session 2 starter — optimizer grouping and learning-rate schedule."""

import torch
import torch.nn as nn


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.Optimizer:
    """Build AdamW groups for matrix weights and vector parameters."""
    # TODO: put trainable parameters with ndim >= 2 in the decay group.
    # TODO: put biases and normalization parameters in the no-decay group.
    # TODO: construct AdamW, optionally using its fused CUDA implementation.
    raise NotImplementedError


def get_lr_cosine_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Return the warmup-plus-cosine learning rate for one optimizer step."""
    # TODO: linearly warm up, cosine-decay, then clamp at min_lr.
    raise NotImplementedError
