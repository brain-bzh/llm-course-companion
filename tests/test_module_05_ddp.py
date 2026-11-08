"""Tests for Module 5 — data parallelism written by hand."""

import pytest
import torch

from nanolm.ddp_manual import allreduce_gradients, allreduce_gradients_flat, broadcast_parameters, shard_batch
from nanolm.model import GPTConfig, MiniGPT


def test_shard_batch_is_a_disjoint_cover():
    batch = torch.arange(12).view(6, 2)
    shards = [shard_batch(batch, rank, 3) for rank in range(3)]
    assert all(s.shape == (2, 2) for s in shards)
    assert torch.equal(torch.cat(shards), batch)


def test_shard_batch_rejects_uneven_split():
    with pytest.raises(ValueError):
        shard_batch(torch.zeros(5, 2), 0, 2)


def test_single_replica_sync_is_a_no_op():
    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16))
    x = torch.randint(0, 32, (2, 9))
    _, loss, _ = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
    loss.backward()
    before = [p.grad.clone() for p in model.parameters()]
    broadcast_parameters(model)
    assert allreduce_gradients(model) == 0
    assert allreduce_gradients_flat(model) == 0
    for grad, expected in zip((p.grad for p in model.parameters()), before):
        assert torch.equal(grad, expected)


@pytest.mark.parametrize("args", [[], ["--accum", "4", "--sync", "flat"]])
def test_two_rank_manual_ddp_matches_global_batch(torchrun, args):
    result = torchrun("05_manual_ddp.py", *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_missing_broadcast_is_detected(torchrun):
    result = torchrun("05_manual_ddp.py", "--skip-broadcast", "--steps", "1")
    assert result.returncode != 0
    assert "FAIL" in result.stdout


def test_two_rank_torch_ddp_with_no_sync_matches_global_batch(torchrun):
    """The library version: DistributedDataParallel with no_sync accumulation."""
    result = torchrun("05_train_ddp.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
