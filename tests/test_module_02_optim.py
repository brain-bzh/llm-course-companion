"""Tests for Module 2, Session 7 — optimizer, schedule and update order.

Checkpointing is Session 8: see test_module_02_baseline.py.
"""

import torch

from nanolm.model import GPTConfig, MiniGPT
from nanolm.optim import configure_optimizers, get_lr_cosine_schedule
from nanolm.train import optimizer_step, train_step


def test_optimizer_parameter_grouping():
    """Verify 2D parameters receive weight decay and 1D biases/norms receive 0.0 decay."""
    config = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32, bias=True)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.1)

    assert len(optimizer.param_groups) == 2
    g_decay = optimizer.param_groups[0]
    g_nodecay = optimizer.param_groups[1]

    assert g_decay["weight_decay"] == 0.1
    assert g_nodecay["weight_decay"] == 0.0

    # Ensure all decay params have dim >= 2
    for p in g_decay["params"]:
        assert p.dim() >= 2

    # Ensure all nodecay params have dim < 2
    for p in g_nodecay["params"]:
        assert p.dim() < 2


def test_cosine_learning_rate_schedule():
    """Verify linear warmup and cosine decay bounds."""
    warmup = 10
    max_steps = 100
    max_lr = 1e-3
    min_lr = 1e-4

    # Warmup start
    lr_0 = get_lr_cosine_schedule(0, warmup, max_steps, max_lr, min_lr)
    assert 0.0 < lr_0 <= max_lr / warmup

    # At warmup end
    lr_warmup = get_lr_cosine_schedule(warmup, warmup, max_steps, max_lr, min_lr)
    assert abs(lr_warmup - max_lr) < 1e-5

    # At max_steps
    lr_end = get_lr_cosine_schedule(max_steps, warmup, max_steps, max_lr, min_lr)
    assert abs(lr_end - min_lr) < 1e-5

    # Beyond max_steps
    lr_beyond = get_lr_cosine_schedule(max_steps + 10, warmup, max_steps, max_lr, min_lr)
    assert abs(lr_beyond - min_lr) < 1e-5


def test_tiny_batch_overfit():
    """The Session 7 training loop must memorize one shifted token sequence."""
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.0, learning_rate=5e-3)
    tokens = torch.tensor([[1, 5, 9, 2, 6, 10, 3, 7, 11]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    loss = float("inf")
    for _ in range(80):
        loss = train_step(model, optimizer, inputs, targets)
        optimizer_step(model, optimizer)

    assert loss < 0.1, f"Model failed to overfit one batch; final loss is {loss:.4f}"
