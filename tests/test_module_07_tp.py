"""Tests for Module 7 — Tensor parallelism."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_tensor_parallel_mlp_equivalence():
    """Verify that a Column + Row parallel MLP is mathematically equivalent to a standard MLP."""
    n_embd = 32
    d_ff = 4 * n_embd

    torch.manual_seed(42)
    ref_fc1 = nn.Linear(n_embd, d_ff, bias=False)
    ref_fc2 = nn.Linear(d_ff, n_embd, bias=False)

    x = torch.randn(2, 8, n_embd)
    ref_out = ref_fc2(F.gelu(ref_fc1(x), approximate="tanh"))

    # Partition across 2 ranks
    w1_r0 = ref_fc1.weight[: d_ff // 2, :]
    w1_r1 = ref_fc1.weight[d_ff // 2 :, :]
    w2_r0 = ref_fc2.weight[:, : d_ff // 2]
    w2_r1 = ref_fc2.weight[:, d_ff // 2 :]

    h_r0 = F.gelu(F.linear(x, w1_r0), approximate="tanh")
    out_r0 = F.linear(h_r0, w2_r0)

    h_r1 = F.gelu(F.linear(x, w1_r1), approximate="tanh")
    out_r1 = F.linear(h_r1, w2_r1)

    # Simulated all-reduce sum
    tp_out = out_r0 + out_r1

    assert torch.allclose(ref_out, tp_out, atol=1e-5)


def test_single_rank_parallel_layers_equal_dense_linear_with_bias():
    """With one rank, the parallel layers must reduce to ordinary nn.Linear (bias included)."""
    from nanolm.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    torch.manual_seed(0)
    dense = nn.Linear(8, 12)
    x = torch.randn(3, 8, requires_grad=True)
    for layer in (ColumnParallelLinear(8, 12, bias=True), RowParallelLinear(8, 12, bias=True)):
        with torch.no_grad():
            layer.weight.copy_(dense.weight)
            layer.bias.copy_(dense.bias)
        x_tp = x.detach().clone().requires_grad_()
        torch.testing.assert_close(layer(x_tp), dense(x))
        layer(x_tp).sum().backward()
        x_ref = x.detach().clone().requires_grad_()
        dense(x_ref).sum().backward()
        torch.testing.assert_close(x_tp.grad, x_ref.grad)


def test_two_rank_sharded_mlp_forward_and_backward(torchrun):
    """Real collectives: outputs, input gradients and both weight shards match."""
    result = torchrun("07_tensor_parallel.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
