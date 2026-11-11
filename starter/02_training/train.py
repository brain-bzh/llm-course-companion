"""Module 2 starter — the training step (Session 7) and checkpoints (Session 8).

Copy over nanolm/train.py.
- Session 7: implement train_step and optimizer_step, then run
      uv run pytest tests/test_module_02_optim.py -q
      uv run python scripts/02_overfit.py
- Session 8: implement save_checkpoint and load_checkpoint (including the RNG
  states), then run
      uv run pytest tests/test_module_02_baseline.py -q
"""

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
    """Session 8: save model, optimizer, configuration, progress and recovery metadata.

    For baseline recovery, also save Python, NumPy and PyTorch RNG states.
    """
    # TODO: build one dictionary with the keys "model" (model.state_dict()),
    #       "optimizer" (optimizer.state_dict()), "config", "step", "val_loss" and
    #       "extra_state" (extra_state or {}). The tests and the baseline read these keys.
    # TODO: add "rng": the Python (random.getstate), NumPy (np.random.get_state)
    #       and PyTorch (torch.get_rng_state) states. Add `import random` and
    #       `import numpy as np` at the top of the file.
    # TODO: write it with torch.save(state, path).
    raise NotImplementedError


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Session 8: restore model/optimizer and optionally the saved RNG states."""
    # TODO: torch.load(path, weights_only=False), then load the model state, and the
    #       optimizer state when an optimizer is given.
    # TODO: if restore_rng, restore the three RNG states saved above (random.setstate,
    #       np.random.set_state, torch.set_rng_state).
    # TODO: return the loaded dictionary so the caller can read step and extra_state.
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
