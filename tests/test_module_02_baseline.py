"""Tests for Module 2, Session 8 — checkpoints and the recoverable baseline."""

from dataclasses import replace

import pytest
import torch

from nanolm.baseline import BaselineConfig, document_manifest, run_baseline
from nanolm.model import GPTConfig, MiniGPT
from nanolm.optim import configure_optimizers
from nanolm.train import load_checkpoint, optimizer_step, save_checkpoint, train_step

TRAIN = ['Training text with a distinct document identity. ' * 3,
         'Another document teaches the model about letters. ' * 3]
VAL = ['Held-out words are measured separately from training. ' * 3]
CFG = BaselineConfig(steps=4, batch_size=2, block_size=8, n_layer=1, n_head=2,
                     n_embd=16, eval_every=2, eval_batches=2)


def test_reject_leakage_and_empty_data():
    with pytest.raises(ValueError, match='Duplicate'):
        document_manifest(['same text'], ['same   text'])
    with pytest.raises(ValueError):
        document_manifest([], VAL)


def test_interrupted_training_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    full = run_baseline(TRAIN, VAL, tmp_path/'full', CFG)
    run_baseline(TRAIN, VAL, tmp_path/'resumed', CFG, stop_after=2)
    resumed = run_baseline(TRAIN, VAL, tmp_path/'resumed', CFG,
                           resume=tmp_path/'resumed/checkpoint.pt')
    assert full['records'] == resumed['records']
    assert resumed['completed_tokens'] == 4*2*8
    a = torch.load(tmp_path/'full/checkpoint.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for param in a['optimizer']['state']:
        for key in a['optimizer']['state'][param]:
            torch.testing.assert_close(a['optimizer']['state'][param][key],
                                       b['optimizer']['state'][param][key], rtol=0, atol=0)


def test_resume_rejects_changed_data_or_schedule(tmp_path):
    run_baseline(TRAIN, VAL, tmp_path, CFG, stop_after=2)
    for docs, cfg in ((VAL + ['new document'], CFG), (VAL, replace(CFG, steps=5))):
        with pytest.raises(ValueError, match='differs'):
            run_baseline(TRAIN, docs, tmp_path, cfg, resume=tmp_path/'checkpoint.pt')


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

    checkpoint_path = tmp_path / "round-trip.pt"
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
