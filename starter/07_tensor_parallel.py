"""Session 21 starter — Megatron-style tensor parallelism.

Copy this file to `nanolm/tensor_parallel.py` on your lab branch, implement the
four TODOs, then run:

    uv run pytest tests/test_module_07_tp.py -q
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/07_tensor_parallel.py

The two autograd functions are the whole communication contract of a
column -> row parallel MLP:

    boundary                       forward          backward
    replicated x -> column shards  identity         SUM all-reduce of dL/dx
    row partial sums -> output     SUM all-reduce   identity
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _check_world(world_size):
    if world_size > 1 and (not dist.is_initialized() or dist.get_world_size() != world_size):
        raise RuntimeError("Initialize a matching TP process group before using a sharded layer")


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, world_size):
        _check_world(world_size)
        ctx.world_size = world_size
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        # TODO: each rank holds only part of dL/dx (from its own output columns).
        #       Return the SUM over TP ranks: clone the contiguous gradient and
        #       all-reduce it when ctx.world_size > 1. Return (grad, None).
        raise NotImplementedError


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, world_size):
        _check_world(world_size)
        # TODO: return the SUM of the partial outputs over TP ranks. Clone x first:
        #       dist.all_reduce works in place. Skip the collective when world_size == 1.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        # Replicated downstream loss supplies the same gradient on every rank.
        # Each input shard receives that gradient once, not another all-reduce.
        return grad, None


class ColumnParallelLinear(nn.Module):
    """Linear layer partitioned along output features (columns of weight matrix).

    Input: [B, ..., in_features] (replicated across all ranks).
    Output: [B, ..., out_features // world_size] (each rank gets its own shard).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        assert out_features % world_size == 0
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each rank computes its slice of the output features
        # TODO: pass x through _CopyToTP.apply(x, self.world_size), then apply
        #       F.linear with this rank's weight and bias slices.
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """Linear layer partitioned along input features (rows of weight matrix).

    Input: [B, ..., in_features // world_size] (sharded across ranks).
    Output: [B, ..., out_features] (summed via all-reduce across all ranks).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: compute the partial product with this rank's weight slice (no bias yet),
        #       sum the partials with _ReduceFromTP.apply, THEN add the bias once.
        # Question: what would go wrong if every rank added the bias before the sum?
        raise NotImplementedError


class ShardedMLP(nn.Module):
    """Toy Tensor-Parallel MLP combining ColumnParallel and RowParallel layers.

    Forward flow:
    x (replicated)
      -> ColumnParallelLinear -> [B, T, 4d/k]
      -> GELU
      -> RowParallelLinear + AllReduce -> [B, T, d] (replicated)
    Notice: No collective communication is required at the intermediate boundary!
    """

    def __init__(self, n_embd: int, world_size: int = 1, rank: int = 0):
        super().__init__()
        self.fc1 = ColumnParallelLinear(n_embd, 4 * n_embd, bias=False, world_size=world_size, rank=rank)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = RowParallelLinear(4 * n_embd, n_embd, bias=False, world_size=world_size, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gelu(self.fc1(x))
        out = self.fc2(h)
        return out


def verify_sharded_mlp():
    """Verify actual forward, input gradient and weight shards on the default group."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(42)
    ref = nn.Sequential(nn.Linear(16, 64, bias=False), nn.GELU(approximate="tanh"),
                        nn.Linear(64, 16, bias=False))
    tp = ShardedMLP(16, world, rank)
    start, stop = rank*(64//world), (rank+1)*(64//world)
    with torch.no_grad():
        tp.fc1.weight.copy_(ref[0].weight[start:stop])
        tp.fc2.weight.copy_(ref[2].weight[:, start:stop])
    x = torch.randn(2, 4, 16, requires_grad=True)
    parallel_x = x.detach().clone().requires_grad_()
    expected, actual = ref(x), tp(parallel_x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for a, b in ((parallel_x.grad, x.grad), (tp.fc1.weight.grad, ref[0].weight.grad[start:stop]),
                 (tp.fc2.weight.grad, ref[2].weight.grad[:, start:stop])):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
