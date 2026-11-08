"""Data parallelism written by hand with three collectives.

Covers:
- Module 5: Distributed data parallelism (batch sharding, replica initialization,
  gradient averaging, communication counting).

`torch.nn.parallel.DistributedDataParallel` performs the same mathematics with
gradient buckets and backward/communication overlap. Writing it by hand first
makes every synchronization point visible.

Every function also works without an initialized process group: it then
behaves as a single replica, so the unit tests can run in one process.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def shard_batch(global_batch: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Return this rank's contiguous, disjoint slice of the global batch (dimension 0).

    Every rank must receive the same number of sequences; otherwise the average
    of per-rank mean losses is no longer the global mean loss.
    """
    if global_batch.size(0) % world_size:
        raise ValueError(f"Global batch {global_batch.size(0)} is not divisible by world size {world_size}")
    local = global_batch.size(0) // world_size
    return global_batch[rank * local : (rank + 1) * local]


@torch.no_grad()
def broadcast_parameters(model: nn.Module, src: int = 0) -> None:
    """Make every replica start from the parameters and buffers of rank `src`."""
    _, world = _world()
    if world == 1:
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.no_grad()
def allreduce_gradients(model: nn.Module) -> int:
    """Average every gradient across ranks, one collective per parameter.

    Returns the number of collectives issued, so the cost can be compared
    with the flat-buffer variant below.
    """
    _, world = _world()
    calls = 0
    for param in model.parameters():  # parameters() yields a tied weight once
        if param.grad is None:
            continue
        if world > 1:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            calls += 1
        param.grad.div_(world)
    return calls


@torch.no_grad()
def allreduce_gradients_flat(model: nn.Module) -> int:
    """Average all gradients with a single collective on one flat buffer.

    This is the idea behind DDP gradient buckets: fewer, larger messages pay
    the per-collective latency once instead of once per parameter.
    """
    _, world = _world()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0
    flat = torch.cat([g.reshape(-1) for g in grads])
    if world > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)
    offset = 0
    for g in grads:
        g.copy_(flat[offset : offset + g.numel()].view_as(g))
        offset += g.numel()
    return 1 if world > 1 else 0


@torch.no_grad()
def max_replica_difference(model: nn.Module) -> float:
    """Largest absolute parameter difference between this replica and rank 0."""
    _, world = _world()
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    if world == 1:
        return 0.0
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    difference = (flat - reference).abs().max()
    dist.all_reduce(difference, op=dist.ReduceOp.MAX)
    return difference.item()
