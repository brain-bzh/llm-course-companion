"""Session 1 — Transformer from first principles.

Exit Criterion:
The model must overfit a tiny batch. If it cannot, the implementation has a bug.
"""

import torch
from minilm.model import MiniGPT, GPTConfig

def main():
    print("=== Session 1: Tiny-Batch Overfitting Test ===")
    config = GPTConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # Tiny synthetic batch (B=1, T=16)
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (1, 16))
    y = torch.randint(0, config.vocab_size, (1, 16))

    print(f"Model parameters: {model.get_num_params():,}")
    print("Beginning 60 optimization steps on fixed tiny batch...")

    initial_loss = None
    for step in range(60):
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1:02d} | Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < 0.1, f"Failed to overfit! Final loss {final_loss} >= 0.1"
    print("SUCCESS: Session 1 exit criterion satisfied (tiny batch overfitted).")

if __name__ == "__main__":
    main()
