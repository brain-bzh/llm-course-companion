"""Tests for Module 2 — Training-loop anatomy."""

import torch

from nanolm.model import GPTConfig, MiniGPT
from nanolm.optim import configure_optimizers, get_lr_cosine_schedule
from nanolm.train import load_checkpoint, optimizer_step, save_checkpoint, train_step


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
    """The Session 2 training loop must memorize one shifted token sequence."""
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, weight_decay=0.0, learning_rate=5e-3)
    tokens = torch.tensor([[1, 5, 9, 2, 6, 10, 3, 7, 11]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    for _ in range(80):
        loss = train_step(model, optimizer, inputs, targets)
        optimizer_step(model, optimizer)

    assert loss < 0.1, f"Model failed to overfit one batch; final loss is {loss:.4f}"


def test_checkpoint_round_trip_preserves_logits(tmp_path):
    """Saving and restoring must preserve model outputs and optimizer state."""
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, learning_rate=1e-3)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    loss = train_step(model, optimizer, inputs, targets)
    optimizer_step(model, optimizer)

    model.eval()
    with torch.no_grad():
        logits_before, _, _ = model(inputs)

    checkpoint_path = tmp_path / "session-02.pt"
    save_checkpoint(checkpoint_path, model, optimizer, config, step=1, val_loss=loss)
    restored_model = MiniGPT(config)
    restored_optimizer = configure_optimizers(restored_model, learning_rate=1e-3)
    metadata = load_checkpoint(checkpoint_path, restored_model, restored_optimizer)

    restored_model.eval()
    with torch.no_grad():
        logits_after, _, _ = restored_model(inputs)

    torch.testing.assert_close(logits_before, logits_after, rtol=0.0, atol=0.0)
    assert metadata["step"] == 1
    assert restored_optimizer.state_dict()["state"]
