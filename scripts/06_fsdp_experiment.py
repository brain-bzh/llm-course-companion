"""Module 6 — FSDP and ZeRO.

Demonstrates:
- Memory scaling comparison: DDP (replicated) vs ZeRO-1 vs ZeRO-2 vs ZeRO-3 (FSDP).
- TransformerBlock auto-wrap policy formulation.
"""

from nanolm.model import GPTConfig, MiniGPT
from nanolm.optim import configure_optimizers
from nanolm.profile_utils import memory_breakdown


def main():
    print("=== Module 6: FSDP & ZeRO Sharding Analysis ===")

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
    z1 = [param_mb + grad_mb + (opt_mb / n) for n in [1, 4, 8, 64]]
    print(f"{'ZeRO-1':<12} | {z1[0]:9.1f} MB | {z1[1]:7.1f} MB | {z1[2]:7.1f} MB | {z1[3]:7.1f} MB")

    # 3. ZeRO-2: Shard optimizer states and gradients across N ranks
    z2 = [param_mb + (grad_mb / n) + (opt_mb / n) for n in [1, 4, 8, 64]]
    print(f"{'ZeRO-2':<12} | {z2[0]:9.1f} MB | {z2[1]:7.1f} MB | {z2[2]:7.1f} MB | {z2[3]:7.1f} MB")

    # 4. ZeRO-3 / FSDP: Shard parameters, gradients, and optimizer states
    z3 = [(param_mb / n) + (grad_mb / n) + (opt_mb / n) for n in [1, 4, 8, 64]]
    print(f"{'FSDP/ZeRO-3':<12} | {z3[0]:9.1f} MB | {z3[1]:7.1f} MB | {z3[2]:7.1f} MB | {z3[3]:7.1f} MB")

    print("\nObservation: FSDP (ZeRO-3) scales down all static states linearly with world size,")
    print("at the cost of requiring all-gather collectives during forward and backward passes.")


if __name__ == "__main__":
    main()
