"""Module 2 capstone: overfit one batch, then (from Session 8) a checkpoint round-trip.

Session 7 implements the update; the script then reports PASS for the overfit and
says that the checkpoint is not implemented yet. After Session 8 implements
save_checkpoint and load_checkpoint, the same script also proves that a fresh model
restored from the checkpoint gives bit-identical logits.
"""

import tempfile
from pathlib import Path

import torch

from nanolm.model import GPTConfig, MiniGPT
from nanolm.optim import configure_optimizers
from nanolm.train import load_checkpoint, optimizer_step, save_checkpoint, train_step


def main() -> None:
    print("=== Module 2: Tiny-Batch Overfit ===")
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.0, learning_rate=5e-3)

    tokens = torch.tensor([[1, 5, 9, 2, 6, 10, 3, 7, 11]])
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    initial_loss: float | None = None
    loss = 0.0
    for step in range(80):
        loss = train_step(model, optimizer, inputs, targets)
        optimizer_step(model, optimizer)
        if initial_loss is None:
            initial_loss = loss
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1:02d} | Loss: {loss:.4f}")

    print(f"Initial loss: {initial_loss:.4f} -> final loss: {loss:.4f}")
    assert loss < 0.1, f"Expected loss < 0.1, got {loss:.4f}"

    print("PASS: the small NanoLM memorizes one batch.")

    model.eval()
    with torch.no_grad():
        logits_before, _, _ = model(inputs)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "overfit.pt")
            save_checkpoint(checkpoint, model, optimizer, config, step=80, val_loss=loss)
            restored_model = MiniGPT(config)
            restored_optimizer = configure_optimizers(restored_model, weight_decay=0.0, learning_rate=5e-3)
            metadata = load_checkpoint(checkpoint, restored_model, restored_optimizer)
    except NotImplementedError:
        print("Checkpoint round-trip: not implemented yet (Session 8).")
        return

    restored_model.eval()
    with torch.no_grad():
        logits_after, _, _ = restored_model(inputs)
    torch.testing.assert_close(logits_before, logits_after, rtol=0.0, atol=0.0)

    print(f"Checkpoint restored at step {metadata['step']} with identical logits.")
    print("PASS: the checkpoint round-trip preserves the model exactly.")


if __name__ == "__main__":
    main()
