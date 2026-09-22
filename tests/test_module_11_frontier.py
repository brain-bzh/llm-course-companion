"""Tests for Module 12 — Frontier architectures."""

import torch
from nanolm.model import GPTConfig
from nanolm.moe import SparseMoELayer


def test_sparse_moe_forward_and_loss():
    config = GPTConfig(vocab_size=100, block_size=32, n_layer=1, n_head=2, n_embd=32)
    moe = SparseMoELayer(config, num_experts=4, top_k=2)

    x = torch.randn(2, 8, config.n_embd)
    out, aux_loss = moe(x)

    assert out.shape == x.shape
    assert aux_loss.item() > 0.0
    assert torch.isfinite(aux_loss)
