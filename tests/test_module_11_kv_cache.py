"""Tests for Module 11 — KV-cached decoding."""

import torch
from minilm.model import MiniGPT, GPTConfig
from minilm.generate import generate_cached, generate_uncached


def test_kv_cached_vs_uncached_exact_match():
    """Verify that cached decoding produces the EXACT identical sequence of tokens as uncached generation."""
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = MiniGPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))

    # Deterministic greedy generation (temp=0.0)
    uncached_out = generate_uncached(model, prompt.clone(), max_new_tokens=15, temperature=0.0)
    cached_out = generate_cached(model, prompt.clone(), max_new_tokens=15, temperature=0.0)

    assert torch.equal(uncached_out, cached_out), "KV-cached output diverged from uncached output!"
