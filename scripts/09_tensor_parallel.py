"""Session 9 — Tensor parallelism.

Demonstrates:
- ColumnParallelLinear: partitioning weight matrix along output columns.
- RowParallelLinear: partitioning weight matrix along input rows with All-Reduce.
- Numerical equivalence check between standard MLP and partitioned MLP.
"""

import torch
import torch.nn as nn
from minilm.model import GPTConfig, MLP
from minilm.tensor_parallel import ColumnParallelLinear, RowParallelLinear

def main():
    print("=== Session 9: Tensor Parallelism Toy Layer ===")
    n_embd = 64
    d_ff = 4 * n_embd
    world_size = 2  # Simulate 2 tensor parallel ranks

    # 1. Standard Reference MLP
    torch.manual_seed(42)
    ref_fc1 = nn.Linear(n_embd, d_ff, bias=False)
    ref_fc2 = nn.Linear(d_ff, n_embd, bias=False)

    x = torch.randn(2, 4, n_embd)
    ref_h = torch.nn.functional.gelu(ref_fc1(x), approximate="tanh")
    ref_out = ref_fc2(ref_h)

    # 2. Sharded Simulation across 2 ranks
    # Rank 0 and Rank 1 slices
    w1_r0 = ref_fc1.weight[: d_ff // 2, :]
    w1_r1 = ref_fc1.weight[d_ff // 2 :, :]

    w2_r0 = ref_fc2.weight[:, : d_ff // 2]
    w2_r1 = ref_fc2.weight[:, d_ff // 2 :]

    # Forward Rank 0
    h_r0 = torch.nn.functional.gelu(torch.nn.functional.linear(x, w1_r0), approximate="tanh")
    out_r0 = torch.nn.functional.linear(h_r0, w2_r0)

    # Forward Rank 1
    h_r1 = torch.nn.functional.gelu(torch.nn.functional.linear(x, w1_r1), approximate="tanh")
    out_r1 = torch.nn.functional.linear(h_r1, w2_r1)

    # Cross-rank All-Reduce sum
    tp_out = out_r0 + out_r1

    max_diff = (ref_out - tp_out).abs().max().item()
    print(f"Column + Row Parallel MLP output shape: {list(tp_out.shape)}")
    print(f"Max absolute difference vs reference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Discrepancy too large: {max_diff}"
    print("SUCCESS: Simulated Tensor Parallel MLP is mathematically identical to reference MLP.")

if __name__ == "__main__":
    main()
