"""Fully Sharded Data Parallel (FSDP2) helpers.

Covers:
- Module 6: FSDP and ZeRO (ZeRO-3 parameter sharding with `fully_shard`, the
  unit lifecycle, per-rank state accounting, and checkpoint formats).

PyTorch's current FSDP API (`torch.distributed.fsdp.fully_shard`, "FSDP2")
replaces each parameter by a DTensor sharded along dimension 0. Calling
`fully_shard` on a submodule makes it one *unit*: its parameters are
all-gathered together before the unit runs and resharded afterwards.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .model import MiniGPT


def apply_fsdp2(model: MiniGPT, mesh, reshard_after_forward: bool = True) -> MiniGPT:
    """Make each Transformer block a unit, then the root (embeddings, final norm, head).

    reshard_after_forward=True  -> ZeRO-3: free gathered weights after each unit's
                                   forward; all-gather them again in backward.
    reshard_after_forward=False -> keep gathered weights from forward until backward
                                   (ZeRO-2-like peak memory, one fewer all-gather).
    """
    from torch.distributed.fsdp import fully_shard

    for block in model.transformer.h:
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


def local_bytes(tensors) -> int:
    """Bytes this rank actually stores for a collection of (possibly sharded) tensors."""
    total = 0
    for t in tensors:
        if t is None:
            continue
        local = t.to_local() if isinstance(t, DTensor) else t
        total += local.numel() * local.element_size()
    return total


def rank_state_bytes(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Measured parameter, gradient and optimizer-state bytes held by this rank."""
    params = list(model.parameters())
    states = [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0]
    return {
        "parameters": local_bytes(params),
        "gradients": local_bytes(p.grad for p in params),
        "optimizer": local_bytes(states),
    }
