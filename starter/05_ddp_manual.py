"""Session 14 starter — data parallelism written by hand.

Copy this file to `nanolm/ddp_manual.py` on your lab branch. Implement the
three core functions, then run:

    uv run pytest tests/test_module_05_ddp.py -q
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py

Every function must also work without an initialized process group (one
replica), so that the unit tests can run in a single process.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def shard_batch(global_batch: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return this rank's contiguous, disjoint slice of the global batch (dimension 0)."""
    # TODO: reject a global batch that is not divisible by world_size (ValueError).
    # TODO: return rows [rank * local, (rank + 1) * local).
    raise NotImplementedError


@torch.no_grad()
def broadcast_parameters(model: nn.Module, src: int = 0) -> None:
    """Make every replica start from the parameters and buffers of rank `src`."""
    # TODO: return immediately for a single replica.
    # TODO: dist.broadcast every parameter and buffer tensor (use .data).
    raise NotImplementedError


@torch.no_grad()
def allreduce_gradients(model: nn.Module) -> int:
    """Average every gradient across ranks, one collective per parameter.

    Return the number of collectives issued (0 for a single replica).
    """
    # TODO: for each parameter with a gradient, SUM-all-reduce it, then divide
    #       by the world size. Skip parameters whose .grad is None.
    # Question: why must every rank visit the parameters in the same order?
    raise NotImplementedError


@torch.no_grad()
def allreduce_gradients_flat(model: nn.Module) -> int:
    """Extension: average all gradients with ONE collective on a flat buffer."""
    # TODO (extension): concatenate the gradients, all-reduce once, divide,
    #       and copy each slice back into its gradient. Return 1 (or 0 alone).
    raise NotImplementedError


@torch.no_grad()
def max_replica_difference(model: nn.Module) -> float:
    """Provided: largest absolute parameter difference with rank 0."""
    _, world = _world()
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    if world == 1:
        return 0.0
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    difference = (flat - reference).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return difference.item()
