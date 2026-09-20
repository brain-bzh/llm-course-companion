"""Module 8 — FSDP and ZeRO.

Demonstrates:
- Memory scaling comparison: DDP (replicated) vs ZeRO-1 vs ZeRO-2 vs ZeRO-3 (FSDP).
- TransformerBlock auto-wrap policy formulation.
"""

from minilm.model import MiniGPT, GPTConfig
from minilm.profile_utils import memory_breakdown
from minilm.optim import configure_optimizers

def main():
    print("=== Module 8: FSDP & ZeRO Sharding Analysis ===")

    # Model representing a medium 350M scale setup
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=24,
        n_head=16,
        n_embd=1024,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model)

    mem = memory_breakdown(model, optimizer)
    param_mb = mem["parameters_mb"]
    grad_mb = mem["gradients_mb"]
    opt_mb = mem["optimizer_states_mb"]

    print(f"Model Configuration: 24 Layers, d_model=1024, ~{model.get_num_params():,} Parameters")
    print("\n--- Theoretical Per-GPU Static Memory vs Cluster World Size (N ranks) ---")
    print(f"{'Strategy':<12} | {'N=1 (Single)':<12} | {'N=4':<10} | {'N=8':<10} | {'N=64':<10}")
    print("-" * 64)

    # 1. DDP (replicates everything)
    ddp_mem = param_mb + grad_mb + opt_mb
    print(f"{'DDP':<12} | {ddp_mem:9.1f} MB | {ddp_mem:7.1f} MB | {ddp_mem:7.1f} MB | {ddp_mem:7.1f} MB")

    # 2. ZeRO-1: Shard optimizer states across N ranks
    for n_ranks in [1, 4, 8, 64]:
        z1 = param_mb + grad_mb + (opt_mb / n_ranks)
        if n_ranks == 1:
            z1_str = f"{z1:9.1f} MB"
        elif n_ranks == 4:
            z1_str_4 = f"{z1:7.1f} MB"
        elif n_ranks == 8:
            z1_str_8 = f"{z1:7.1f} MB"
        elif n_ranks == 64:
            z1_str_64 = f"{z1:7.1f} MB"
    print(f"{'ZeRO-1':<12} | {z1_str} | {z1_str_4} | {z1_str_8} | {z1_str_64}")

    # 3. ZeRO-2: Shard optimizer states and gradients across N ranks
    for n_ranks in [1, 4, 8, 64]:
        z2 = param_mb + (grad_mb / n_ranks) + (opt_mb / n_ranks)
        if n_ranks == 1:
            z2_str = f"{z2:9.1f} MB"
        elif n_ranks == 4:
            z2_str_4 = f"{z2:7.1f} MB"
        elif n_ranks == 8:
            z2_str_8 = f"{z2:7.1f} MB"
        elif n_ranks == 64:
            z2_str_64 = f"{z2:7.1f} MB"
    print(f"{'ZeRO-2':<12} | {z2_str} | {z2_str_4} | {z2_str_8} | {z2_str_64}")

    # 4. ZeRO-3 / FSDP: Shard parameters, gradients, and optimizer states
    for n_ranks in [1, 4, 8, 64]:
        z3 = (param_mb / n_ranks) + (grad_mb / n_ranks) + (opt_mb / n_ranks)
        if n_ranks == 1:
            z3_str = f"{z3:9.1f} MB"
        elif n_ranks == 4:
            z3_str_4 = f"{z3:7.1f} MB"
        elif n_ranks == 8:
            z3_str_8 = f"{z3:7.1f} MB"
        elif n_ranks == 64:
            z3_str_64 = f"{z3:7.1f} MB"
    print(f"{'FSDP/ZeRO-3':<12} | {z3_str} | {z3_str_4} | {z3_str_8} | {z3_str_64}")

    print("\nObservation: FSDP (ZeRO-3) scales down all static states linearly with world size,")
    print("at the cost of requiring all-gather collectives during forward and backward passes.")

if __name__ == "__main__":
    main()
