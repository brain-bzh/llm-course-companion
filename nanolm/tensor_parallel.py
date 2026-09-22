"""Megatron-style Tensor Parallelism (TP).

Covers:
- Module 9: Tensor parallelism (ColumnParallelLinear, RowParallelLinear, toy sharded MLP).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


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
        return F.linear(x, self.weight, self.bias)


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
        output_parallel = F.linear(x, self.weight)

        # Cross-rank All-Reduce sum
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        return output_parallel


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
