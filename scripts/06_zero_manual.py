"""Module 6: hand-written ZeRO sharded AdamW checked against one global batch.

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_zero_manual.py
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=4 scripts/06_zero_manual.py --clip 0.5

Each rank holds a full model replica (as in DDP) but AdamW moments only for
its own slice of the flattened parameters. The reference is one process
training on the whole global batch with torch.optim.AdamW.
"""

import argparse
import os

import torch
import torch.distributed as dist

from nanolm.distributed import configure_local_gloo
from nanolm.ddp_manual import max_replica_difference, shard_batch
from nanolm.model import GPTConfig, MiniGPT
from nanolm.zero import ShardedAdamW


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--clip", type=float, default=None, help="extension: global gradient-norm clipping")
    args = parser.parse_args()
    torch.set_num_threads(1)
    configure_local_gloo()
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    log = print if rank == 0 else (lambda *a, **k: None)
    try:
        cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
        torch.manual_seed(0)  # identical initialization everywhere
        model = MiniGPT(cfg)
        reference = MiniGPT(cfg)
        reference.load_state_dict(model.state_dict())
        hyper = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        optimizer = ShardedAdamW(model.parameters(), max_grad_norm=args.clip, **hyper)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), **hyper)

        n = optimizer.numel
        padding = optimizer.shard_size * world - n
        log(f"parameters N={n:,}  world={world}  shard_size={optimizer.shard_size:,}  padding={padding}")
        log(f"AdamW moments per rank: DDP {8 * n:,} B  ->  sharded {optimizer.state_bytes():,} B "
            f"(prediction 8 * shard_size = {8 * optimizer.shard_size:,} B)")

        data = torch.Generator().manual_seed(123)
        for step in range(1, args.steps + 1):
            tokens = torch.randint(0, cfg.vocab_size, (2 * world, 9), generator=data)
            local = shard_batch(tokens, rank, world)
            optimizer.zero_grad()
            _, loss, _ = model(local[:, :-1].contiguous(), targets=local[:, 1:].contiguous())
            loss.backward()
            optimizer.step()

            ref_optimizer.zero_grad(set_to_none=True)
            _, ref_loss, _ = reference(tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
            ref_loss.backward()
            ref_norm = None
            if args.clip is not None:
                ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), args.clip).item()
            ref_optimizer.step()

            error = max((a - b).abs().max().item() for a, b in zip(model.parameters(), reference.parameters()))
            drift = max_replica_difference(model)
            norm_text = ""
            if args.clip is not None:
                norm_text = f"  grad-norm {optimizer.last_grad_norm:.4f} vs reference {ref_norm:.4f}"
            log(f"step {step}: |param - reference|={error:.2e}  replica drift={drift:.2e}{norm_text}")
            ok = torch.tensor(float(error < 5e-5 and drift == 0.0))
            if dist.is_initialized():
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() < 1:
                log("FAIL: sharded update differs from torch.optim.AdamW on the global batch.")
                raise SystemExit(1)
        log(f"PASS: {world} ranks with sharded AdamW state match one global-batch AdamW run.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
