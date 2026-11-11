"""Tests for Module 6 — sharded optimizer state, FSDP2, and checkpoint formats."""

import pytest
import torch

from nanolm.model import GPTConfig, MiniGPT
from nanolm.zero import ShardedAdamW


def _model():
    torch.manual_seed(0)
    return MiniGPT(GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16))


def test_single_rank_sharded_adamw_equals_torch_adamw():
    model, reference = _model(), _model()
    hyper = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer = ShardedAdamW(model.parameters(), **hyper)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), **hyper)
    x = torch.randint(0, 32, (4, 9), generator=torch.Generator().manual_seed(1))
    for _ in range(3):
        for m, opt in ((model, optimizer), (reference, ref_optimizer)):
            opt.zero_grad()
            _, loss, _ = m(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
            loss.backward()
            opt.step()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    assert optimizer.state_bytes() == 8 * optimizer.numel


def test_tied_weights_are_counted_once():
    model = _model()
    optimizer = ShardedAdamW(model.parameters())
    assert optimizer.numel == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("nproc", [2, 3])
def test_sharded_adamw_matches_torch_adamw_across_ranks(torchrun, nproc):
    """Three ranks do not divide the parameter count: the last shard is padded."""
    result = torchrun("06_zero_manual.py", nproc=nproc)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


@pytest.mark.extension
def test_two_rank_sharded_adamw_with_global_clipping(torchrun):
    result = torchrun("06_zero_manual.py", "--clip", "0.5")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_fsdp2_matches_reference_and_checkpoint_reshards(torchrun, tmp_path):
    result = torchrun("06_fsdp2.py", "--save", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
    assert "DTensor" in result.stdout and "Parameter" in result.stdout

    import subprocess
    import sys
    from conftest import ROOT

    reload = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_checkpoint_formats.py"), str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert reload.returncode == 0, reload.stdout + reload.stderr
    assert "PASS:" in reload.stdout
