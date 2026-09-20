"""Module 13 — Beyond dense Transformers.

Demonstrates:
- Sparse Mixture of Experts (MoE) layer with top-k router.
- Auxiliary load-balancing loss calculation.
- Parameter vs compute scaling trade-off (more parameters without multiplying FLOPs).
"""

import torch
from minilm.model import GPTConfig, MLP
from minilm.moe import SparseMoELayer

def main():
    print("=== Module 13: Beyond Dense Transformers (Sparse MoE) ===")
    config = GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=128,
    )

    num_experts = 4
    top_k = 2
    moe_layer = SparseMoELayer(config, num_experts=num_experts, top_k=top_k)
    dense_mlp = MLP(config)

    dense_params = sum(p.numel() for p in dense_mlp.parameters())
    moe_params = sum(p.numel() for p in moe_layer.parameters())

    print(f"Dense MLP Parameters:  {dense_params:,}")
    print(f"Sparse MoE Parameters:  {moe_params:,} ({num_experts} experts, top-{top_k} routing)")
    print(f"Parameter expansion:    {moe_params / dense_params:.2f}x capacity")
    print(f"Active compute ratio:   {top_k / num_experts:.2f}x active FLOPs (only top-{top_k} run per token)")

    # Forward pass
    x = torch.randn(2, 16, config.n_embd)
    out, aux_loss = moe_layer(x)

    print(f"\nInput shape:        {list(x.shape)}")
    print(f"Output shape:       {list(out.shape)}")
    print(f"Aux Load-Balancing Loss: {aux_loss.item():.4f}")
    assert out.shape == x.shape, "Shape mismatch in MoE forward output!"
    print("SUCCESS: Sparse MoE layer executed with correct shapes and load balancing.")

if __name__ == "__main__":
    main()
