"""Define a logical training layout; optionally revisit DDP and TP with Gloo.

Run the default logical layout with:
    uv run python scripts/08_parallelism_layout.py

For the real two-rank check, set LAYOUT to a pure TP=2 layout below and run:
    uv run torchrun --standalone --nproc_per_node=2 scripts/08_parallelism_layout.py --gloo-check
"""
import argparse
from itertools import product
import os
import sys

import torch
import torch.distributed as dist

from nanolm.model import GPTConfig, MiniGPT
from nanolm.parallelism_calc import (
    ModelSpecs,
    ParallelismConfig,
    calculate_memory_and_scaling,
)


# Student exercise: change the model and layout, then explain the printed groups.
MODEL = ModelSpecs(
    name="Teaching MoE",
    n_layer=8,
    n_head=8,
    d_model=512,
    seq_len=2048,
    vocab_size=32000,
    n_experts=8,
    top_k=2,
)
LAYOUT = ParallelismConfig(
    dp_size=4,
    tp_size=2,
    pp_size=2,
    cp_size=2,
    ep_size=4,
    fsdp_size=1,
    microbatches=8,
)
BATCH_SIZE_PER_DP_RANK = 1
ACTIVATION_DTYPE_BYTES = 2  # bf16/fp16 payload estimate


AXES = ("dp", "tp", "pp", "cp")


def rank_id(coordinates, sizes):
    """Map (dp, tp, pp, cp) coordinates to a row-major logical rank."""
    rank = 0
    for axis in AXES:
        rank = rank * sizes[axis] + coordinates[axis]
    return rank


def rank_coordinates(rank, sizes):
    coordinates = {}
    for axis in reversed(AXES):
        coordinates[axis] = rank % sizes[axis]
        rank //= sizes[axis]
    return coordinates


def groups_for_axis(axis, sizes):
    """Return logical groups that vary along one axis and hold others fixed."""
    varying_size = sizes[axis]
    other_axes = [name for name in AXES if name != axis]
    groups = []
    for fixed in product(*(range(sizes[name]) for name in other_axes)):
        base = dict(zip(other_axes, fixed))
        groups.append([
            rank_id({**base, axis: index}, sizes)
            for index in range(varying_size)
        ])
    return groups


def print_rank_layout(specs, config):
    sizes = {
        "dp": config.dp_size,
        "tp": config.tp_size,
        "pp": config.pp_size,
        "cp": config.cp_size,
    }
    world_size = 1
    for size in sizes.values():
        world_size *= size

    print(f"Logical mesh (dp, tp, pp, cp): {tuple(sizes[a] for a in AXES)}")
    print(f"Logical devices: {world_size}")
    layers_per_rank = specs.n_layer // config.pp_size
    tokens_per_rank = specs.seq_len // config.cp_size
    print(f"Layers/PP rank: {layers_per_rank}; tokens/CP rank: {tokens_per_rank}")
    print(f"Attention heads/TP rank: {specs.n_head // config.tp_size}")
    print("Rank coordinates (rank: dp,tp,pp,cp):")
    for rank in range(world_size):
        coord = rank_coordinates(rank, sizes)
        print(f"  {rank}: " + ",".join(f"{axis}={coord[axis]}" for axis in AXES))

    for axis in AXES:
        if sizes[axis] > 1:
            groups = groups_for_axis(axis, sizes)
            print(f"{axis.upper()} groups: {groups}")

    for name, size in (("FSDP", config.fsdp_size), ("EP", config.ep_size)):
        if size == 1:
            continue
        dp_groups = groups_for_axis("dp", sizes)
        shard_groups = [group[i:i + size] for group in dp_groups for i in range(0, len(group), size)]
        replicas = [
            [group[i] for i in range(0, len(group), size)]
            for group in dp_groups
            for i in range(size)
        ]
        print(f"{name} subgroups within each DP group: {shard_groups}")
        print(f"{name} replica groups across subdivisions: {replicas}")


def print_meta_model_summary(specs, tp_size):
    """Use NanoLM on meta to inspect dense and TP-sharded shapes without weight data."""
    config = GPTConfig(
        vocab_size=specs.vocab_size,
        block_size=specs.seq_len,
        n_layer=specs.n_layer,
        n_head=specs.n_head,
        n_embd=specs.d_model,
    )
    with torch.device("meta"):
        model = MiniGPT(config)

    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    print("NanoLM meta census (dense model; no parameter data allocated):")
    print(f"  unique parameters: {total:,}")
    print("  largest parameter tensors:")
    for name, parameter in sorted(parameters, key=lambda item: item[1].numel(), reverse=True)[:5]:
        print(f"    {name}: {tuple(parameter.shape)} ({parameter.numel():,} values)")
    column_weight = model.transformer.h[0].mlp.c_fc.weight
    row_weight = model.transformer.h[0].mlp.c_proj.weight
    if column_weight.shape[0] % tp_size or row_weight.shape[1] % tp_size:
        raise ValueError("MLP dimensions must divide evenly across the selected TP size.")
    column_shards = column_weight.chunk(tp_size, dim=0)
    row_shards = row_weight.chunk(tp_size, dim=1)
    print(f"  TP={tp_size} example (meta shapes only):")
    print(
        f"    column-parallel c_fc: {tuple(column_weight.shape)} "
        f"-> {tuple(column_shards[0].shape)} per rank"
    )
    print(
        f"    row-parallel c_proj: {tuple(row_weight.shape)} "
        f"-> {tuple(row_shards[0].shape)} per rank"
    )
    if specs.n_experts > 1:
        print(f"  MoE experts ({specs.n_experts}, top-{specs.top_k}) are estimated separately by the sizing model.")


def print_cost_summary(specs, config):
    result = calculate_memory_and_scaling(
        specs,
        config,
        batch_size_per_gpu=BATCH_SIZE_PER_DP_RANK,
        dtype_bytes=ACTIVATION_DTYPE_BYTES,
    )
    print("Idealized sizing (not peak VRAM or a speed prediction):")
    print(f"  devices: {result['total_gpus']}")
    print(f"  global tokens/update: {result['global_tokens_per_step']:,}")
    print(f"  persistent state/rank: {result['persistent_state_bytes'] / 2**30:.3f} GiB")
    print(f"  pipeline idle fraction: {100 * result['pipeline_bubble_fraction']:.1f}%")
    print(f"  excluded: {result['exclusions']}")
    local_tokens = (
        BATCH_SIZE_PER_DP_RANK
        * (specs.seq_len // config.cp_size)
        * config.microbatches
    )
    print("Illustrative communication payloads (bytes, no latency or overlap model):")
    if config.pp_size > 1:
        pp_payload = (
            BATCH_SIZE_PER_DP_RANK
            * (specs.seq_len // config.cp_size)
            * specs.d_model
            * ACTIVATION_DTYPE_BYTES
        )
        print(f"  PP forward activation per microbatch per boundary (aggregate across TP): {pp_payload:,}")
    if config.ep_size > 1:
        remote_fraction = 1 - 1 / config.ep_size
        ep_payload = (
            local_tokens
            * specs.top_k
            * specs.d_model
            * ACTIVATION_DTYPE_BYTES
            * remote_fraction
        )
        print(
            f"  EP remote dispatch sent/routing-rank/update: {ep_payload:,.0f} "
            f"(uniform routing, full-width hidden states; excludes TP sharding, "
            "return, backward, padding, and metadata)"
        )


def run_gloo_reuse_check(config):
    if not all((
        config.tp_size > 1,
        config.dp_size == 1,
        config.pp_size == 1,
        config.cp_size == 1,
        config.fsdp_size == 1,
        config.ep_size == 1,
    )):
        raise ValueError("The Gloo check requires a pure TP layout; set TP to the process count and other axes to 1.")
    if "WORLD_SIZE" not in os.environ:
        raise ValueError("Launch the Gloo check with torchrun so multiple CPU ranks participate.")
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

    dist.init_process_group("gloo")
    try:
        if dist.get_world_size() != config.tp_size:
            raise ValueError(
                f"TP size {config.tp_size} does not match torchrun world size {dist.get_world_size()}."
            )
        from nanolm.distributed import reduce_tensor
        from nanolm.tensor_parallel import verify_sharded_mlp

        rank = dist.get_rank()
        averaged = reduce_tensor(torch.tensor([float(rank + 1)]), average=True)
        torch.testing.assert_close(averaged, torch.tensor([1.5]))
        if rank == 0:
            print("PASS: reused DDP all-reduce helper returns the expected global average.")
        verify_sharded_mlp()
        if dist.get_rank() == 0:
            print("PASS: Gloo TP outputs, input gradients, and weight shards match the reference.")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gloo-check",
        action="store_true",
        help="run reused DDP all-reduce and TP equivalence checks on real CPU tensors",
    )
    args = parser.parse_args()

    # Reuse the calculator's divisibility and EP/FSDP compatibility checks.
    calculate_memory_and_scaling(
        MODEL,
        LAYOUT,
        batch_size_per_gpu=BATCH_SIZE_PER_DP_RANK,
        dtype_bytes=ACTIVATION_DTYPE_BYTES,
    )
    print_rank_layout(MODEL, LAYOUT)
    print_meta_model_summary(MODEL, LAYOUT.tp_size)
    print_cost_summary(MODEL, LAYOUT)
    if args.gloo_check:
        run_gloo_reuse_check(LAYOUT)


if __name__ == "__main__":
    main()
