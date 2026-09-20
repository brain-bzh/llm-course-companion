"""Module 2 — Training-loop anatomy.

Demonstrates:
- Decoupled weight decay parameter grouping.
- Warmup and cosine decay learning rate schedule.
- Gradient accumulation and norm clipping.
- Checkpoint saving and resumption.
"""

import tempfile
import torch
from minilm.model import MiniGPT, GPTConfig
from minilm.optim import configure_optimizers, get_lr_cosine_schedule
from minilm.train import train_step, optimizer_step, save_checkpoint, load_checkpoint

def main():
    print("=== Module 2: Training Loop Anatomy ===")
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, bias=True)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-3)

    # Inspect parameter groups
    g0_count = sum(p.numel() for p in optimizer.param_groups[0]["params"])
    g1_count = sum(p.numel() for p in optimizer.param_groups[1]["params"])
    print(f"Decayed parameters (2D weights): {g0_count:,}")
    print(f"Non-decayed parameters (biases / norms): {g1_count:,}")

    # Simulate 5 steps with gradient accumulation of 2
    grad_accum_steps = 2
    for step in range(5):
        lr = get_lr_cosine_schedule(step, warmup_steps=2, max_steps=10, max_lr=1e-3)
        for g in optimizer.param_groups:
            g["lr"] = lr

        for micro in range(grad_accum_steps):
            x = torch.randint(0, config.vocab_size, (2, 16))
            y = torch.randint(0, config.vocab_size, (2, 16))
            loss = train_step(model, optimizer, x, y, grad_accum_steps=grad_accum_steps)

        norm = optimizer_step(model, optimizer, grad_clip=1.0)
        print(f"Step {step+1} | LR: {lr:.6f} | Micro-Loss: {loss:.4f} | Grad Norm: {norm:.4f}")

    # Verify Checkpointing
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        save_checkpoint(tmp.name, model, optimizer, config, step=5, val_loss=1.23)
        model2 = MiniGPT(config)
        meta = load_checkpoint(tmp.name, model2)
        print(f"\nCheckpoint successfully saved & recovered. Step: {meta['step']}, Val loss: {meta['val_loss']}")

if __name__ == "__main__":
    main()
