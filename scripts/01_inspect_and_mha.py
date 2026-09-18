"""Session 1 — Inspect Transformer and Reimplement Naive MHA.

Covers:
- Part 1: Inspect reference Transformer parameters, tensor dimensions, and state dict.
- Part 2: Explicit NaiveMultiHeadAttention implementation from first principles.
- Part 3: Shape validation and causal masking invariance test.
- Part 4: Tiny-batch overfitting exit criterion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from minilm.model import MiniGPT, GPTConfig


class NaiveMultiHeadAttention(nn.Module):
    """Pedagogically explicit Multi-Head Attention without helper abstractions."""

    def __init__(self, d_model: int, n_head: int, block_size: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # 1. Individual projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 2. Lower-triangular causal mask
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Step 1: Linear projections
        q = self.w_q(x)  # [B, T, d_model]
        k = self.w_k(x)  # [B, T, d_model]
        v = self.w_v(x)  # [B, T, d_model]

        # Step 2: Split into heads and transpose
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, T, d_head]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, T, d_head]
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, n_head, T, d_head]

        # Step 3: Scaled dot-product scores
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))  # [B, n_head, T, T]

        # Step 4: Causal masking (mask future tokens)
        causal_mask = self.mask[:, :, :T, :T] == 0
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Step 5: Softmax & weighted values product
        attn_weights = F.softmax(scores, dim=-1)
        out_heads = attn_weights @ v  # [B, n_head, T, d_head]

        # Step 6: Concatenate heads and project
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, C)
        output = self.w_o(out_concat)
        return output


def main():
    print("=== Session 1: Inspect Transformer & Naive MHA ===")

    # -------------------------------------------------------------
    # Part 1: Inspect Reference Transformer
    # -------------------------------------------------------------
    print("\n[Part 1] Model parameter inventory:")
    config = GPTConfig(vocab_size=1000, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = MiniGPT(config)

    for name, param in list(model.named_parameters())[:8]:
        print(f"  {name:<38} | Shape: {str(list(param.shape)):<18} | Count: {param.numel():,}")
    print(f"  ... total parameters: {model.get_num_params():,}")

    # -------------------------------------------------------------
    # Part 2 & 3: Naive MHA Shape & Causal Invariance Check
    # -------------------------------------------------------------
    print("\n[Part 2 & 3] Naive MHA Invariant Checks:")
    mha = NaiveMultiHeadAttention(d_model=64, n_head=4, block_size=32)
    mha.eval()

    # Shape check
    x = torch.randn(2, 16, 64)
    y = mha(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Shape check passed: Input {list(x.shape)} -> Output {list(y.shape)}")

    # Causal invariance check
    seq1 = torch.randn(1, 6, 64)
    seq2 = seq1.clone()
    seq2[:, 4:, :] = torch.randn(1, 2, 64)  # Mutate future tokens at pos 4 and 5

    with torch.no_grad():
        out1 = mha(seq1)
        out2 = mha(seq2)

    diff_prefix = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
    print(f"  Causal invariance verified! Max diff at prefix positions (0..3): {diff_prefix:.2e}")
    assert diff_prefix < 1e-6, f"Causality leak detected: diff={diff_prefix}"

    # -------------------------------------------------------------
    # Part 4: Tiny-Batch Overfit (Exit Criterion)
    # -------------------------------------------------------------
    print("\n[Part 4] Tiny-Batch Overfit Test:")
    torch.manual_seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    idx = torch.randint(0, config.vocab_size, (1, 16))
    targets = torch.randint(0, config.vocab_size, (1, 16))

    for step in range(60):
        optimizer.zero_grad()
        _, loss, _ = model(idx, targets=targets)
        loss.backward()
        optimizer.step()
        if (step + 1) % 15 == 0:
            print(f"  Step {step+1:02d} | Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print(f"Final loss: {final_loss:.4f}")
    assert final_loss < 0.1, f"Failed exit criterion: final loss {final_loss} >= 0.1"
    print("\nSUCCESS: All Session 1 checks and exit criteria satisfied!")


if __name__ == "__main__":
    main()
