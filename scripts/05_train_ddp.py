"""Module 6 — Distributed data parallelism (DDP).

Can be executed standalone or via torchrun:
    torchrun --nproc_per_node=2 scripts/07_train_ddp.py

Demonstrates:
- Distributed environment setup (rank, local_rank, world_size).
- DistributedDataParallel model wrapping.
- Gradient accumulation and cross-rank all-reduce.
- Global token accounting.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nanolm.model import MiniGPT, GPTConfig
from nanolm.optim import configure_optimizers
from nanolm.distributed import setup_distributed, cleanup_distributed, global_token_count

def main():
    is_dist, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"=== Module 6: Distributed Data Parallelism ===")
        print(f"World Size: {world_size} | Distributed: {is_dist}")

    config = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = MiniGPT(config).to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = configure_optimizers(model, learning_rate=1e-3)

    # Local training step with rank-specific data
    torch.manual_seed(42 + rank)
    local_batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (local_batch_size, seq_len), device=device)
    y = torch.randint(0, config.vocab_size, (local_batch_size, seq_len), device=device)

    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()

    local_tokens = local_batch_size * seq_len
    global_tokens = global_token_count(local_tokens)

    if rank == 0:
        print(f"Rank 0 Step Complete | Local Loss: {loss.item():.4f}")
        print(f"Global Tokens Processed this Step: {global_tokens} (Local: {local_tokens} x {world_size} ranks)")

    cleanup_distributed()

if __name__ == "__main__":
    main()
