"""Fully Sharded Data Parallel (FSDP) and ZeRO utilities.

Covers:
- Module 8: FSDP and ZeRO (sharding strategy, auto-wrap policy, activation checkpointing).
"""

from typing import Optional
import functools
import torch
import torch.nn as nn
from .model import TransformerBlock


def get_fsdp_wrap_policy():
    """Create a module-based auto-wrap policy that wraps each TransformerBlock individually."""
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
    except ImportError:
        return None


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy_str: str = "FULL_SHARD",
    use_activation_checkpointing: bool = False,
) -> nn.Module:
    """Wrap a model in PyTorch FullyShardedDataParallel (FSDP).

    Supported strategies:
    - FULL_SHARD: Shard parameters, gradients, and optimizer states (ZeRO-3).
    - SHARD_GRAD_OP: Shard gradients and optimizer states (ZeRO-2).
    - NO_SHARD: Traditional DDP replication.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategies.get(sharding_strategy_str, ShardingStrategy.FULL_SHARD)
    auto_wrap_policy = get_fsdp_wrap_policy()

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )

    if use_activation_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
        )

    return fsdp_model
