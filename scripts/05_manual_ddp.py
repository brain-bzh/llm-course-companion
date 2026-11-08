"""Module 5: hand-written data parallelism checked against one global batch.

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py --accum 4 --sync flat
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py --skip-broadcast

Each rank deliberately initializes its replica with a DIFFERENT seed. Only
`broadcast_parameters` can make the replicas identical, and only
`allreduce_gradients` keeps them identical after AdamW updates. A
single-process model trained on the whole global batch is the reference.

This is a correctness experiment on synthetic tokens, not a scaling benchmark.
"""

import argparse
import os

import torch
import torch.distributed as dist

from nanolm.distributed import configure_local_gloo
from nanolm.ddp_manual import (
    allreduce_gradients,
    allreduce_gradients_flat,
    broadcast_parameters,
    max_replica_difference,
    shard_batch,
)
from nanolm.model import GPTConfig, MiniGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accum", type=int, default=2, help="micro-batches per rank per update")
    parser.add_argument("--micro-batch", type=int, default=2, help="sequences per micro-batch")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sync", choices=["per-param", "flat"], default="per-param")
    parser.add_argument("--skip-broadcast", action="store_true", help="failure demo: replicas start differently")
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
        torch.manual_seed(1000 + rank)  # different on purpose
        model = MiniGPT(cfg)
        log(f"before broadcast: max replica difference = {max_replica_difference(model):.3e}")
        if not args.skip_broadcast:
            broadcast_parameters(model)
        log(f"after broadcast:  max replica difference = {max_replica_difference(model):.3e}")

        reference = MiniGPT(cfg)
        reference.load_state_dict(model.state_dict())  # rank-local copy of the starting point
        hyper = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), **hyper)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), **hyper)
        sync = allreduce_gradients if args.sync == "per-param" else allreduce_gradients_flat
        n_params = sum(p.numel() for p in model.parameters())
        global_rows = world * args.accum * args.micro_batch
        log(f"world={world} accum={args.accum} micro_batch={args.micro_batch} seq_len=8 "
            f"-> global batch {global_rows} sequences = {global_rows * 8} tokens/update")

        data = torch.Generator().manual_seed(123)  # same global batches on every rank
        for step in range(1, args.steps + 1):
            tokens = torch.randint(0, cfg.vocab_size, (global_rows, 9), generator=data)
            local = shard_batch(tokens, rank, world)

            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.accum):  # no communication inside the accumulation loop
                rows = local[micro * args.micro_batch : (micro + 1) * args.micro_batch]
                _, loss, _ = model(rows[:, :-1].contiguous(), targets=rows[:, 1:].contiguous())
                (loss / args.accum).backward()
            calls = sync(model)
            optimizer.step()

            ref_optimizer.zero_grad(set_to_none=True)
            _, ref_loss, _ = reference(tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
            ref_loss.backward()
            ref_optimizer.step()

            param_error = max((a - b).abs().max().item() for a, b in zip(model.parameters(), reference.parameters()))
            drift = max_replica_difference(model)
            log(f"step {step}: collectives={calls:3d}  sent≈{2 * (world - 1) / world * 4 * n_params / 1e3:6.1f} kB/rank "
                f"(ring estimate)  |param - reference|={param_error:.2e}  replica drift={drift:.2e}")
            ok = param_error < 1e-5 and drift == 0.0
            flag = torch.tensor(float(ok))
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() < 1:
                log("FAIL: replicas diverge from the global-batch reference. Check broadcast, "
                    "loss normalization and gradient averaging.")
                raise SystemExit(1)
        log(f"PASS: {world} hand-written replicas match one global-batch AdamW run "
            f"(sync={args.sync}, accumulation={args.accum}).")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
