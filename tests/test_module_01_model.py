"""Tests for Module 1 — Transformer from first principles."""

import pytest
import torch
from minilm.model import MiniGPT, GPTConfig


def test_model_output_shape():
    config = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss, _ = model(x)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is None


def test_causal_masking_invariance():
    """Verify strictly that future tokens cannot influence earlier token logits."""
    config = GPTConfig(
        vocab_size=256,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_sdpa=False,  # Use manual masked attention to explicitly verify causal mask logic
    )
    model = MiniGPT(config)
    model.eval()

    # Sequence 1
    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    # Sequence 2: identical prefix, but different future tokens at pos 3 and 4
    seq2 = torch.tensor([[10, 20, 30, 99, 88]])

    with torch.no_grad():
        logits1, _, _ = model(seq1)
        logits2, _, _ = model(seq2)

    # Logits at position 0, 1, 2 must be EXACTLY identical between seq1 and seq2
    diff_prefix = (logits1[:, :3, :] - logits2[:, :3, :]).abs().max().item()
    assert diff_prefix < 1e-5, f"Causality leak! Future tokens altered earlier logits: {diff_prefix}"

    # Logits at position 3 and 4 should differ
    diff_suffix = (logits1[:, 3:, :] - logits2[:, 3:, :]).abs().max().item()
    assert diff_suffix > 1e-4, "Expected different logits at modified positions"


def test_tiny_batch_overfit():
    """Module 1 Exit Criterion: model must overfit a tiny batch."""
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    x = torch.randint(0, config.vocab_size, (1, 8))
    y = torch.randint(0, config.vocab_size, (1, 8))

    for _ in range(60):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"Model failed to overfit tiny batch, loss is {loss.item():.4f}"
