"""Distributed Data Parallelism (DDP) helpers.

Covers:
- Module 5: Distributed data parallelism (setup, all-reduce, rank synchronization, token accounting).
"""

import os
import sys
from typing import Optional, Tuple
import torch
import torch.distributed as dist


_LOOPBACK = {"linux": "lo", "darwin": "lo0"}


def configure_local_gloo() -> None:
    """Pin Gloo to the loopback interface for single-machine CPU exercises.

    Gloo picks a network interface by resolving the hostname. On laptops this
    can select an unreachable address (VPN, Wi-Fi, IPv6) and hang. When every
    rank runs on this machine, the loopback interface is always correct.

    - Linux: `lo`; macOS: `lo0`; Windows: left to Gloo's default.
    - Never overrides a value you exported yourself.
    - Does nothing for multi-node jobs (LOCAL_WORLD_SIZE != WORLD_SIZE).

    Call it in each worker before `dist.init_process_group("gloo")`.
    """
    world = os.environ.get("WORLD_SIZE")
    if world is not None and os.environ.get("LOCAL_WORLD_SIZE", world) != world:
        return
    interface = _LOOPBACK.get(sys.platform)
    if interface is not None:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", interface)


def setup_distributed() -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed if running under torchrun / MPI.

    Returns:
        (is_distributed, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "gloo":
            configure_local_gloo()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend)
        return True, rank, local_rank, world_size

    return False, 0, 0, 1


def cleanup_distributed() -> None:
    """Destroy distributed process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce a tensor across all distributed workers."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t /= dist.get_world_size()
    return t


def global_token_count(local_tokens: int) -> int:
    """Sum token count across all ranks for verified global accounting."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_tokens
    t = torch.tensor([local_tokens], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())
