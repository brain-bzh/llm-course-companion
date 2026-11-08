"""Module 6: real FSDP2 (ZeRO-3) on CPU ranks — lifecycle, equivalence, memory.

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py --keep-after-forward
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py --save runs/fsdp2-ckpt

Prints (rank 0):
1. the parameter lifecycle of each unit during the first step: sharded DTensor
   at rest, gathered full tensor while the unit runs;
2. the measured bytes this rank holds for parameters, gradients and AdamW
   state, next to the replicated (DDP) values;
3. agreement with one process training on the whole global batch.

With --save DIR, writes a full (rank-0) checkpoint and a sharded distributed
checkpoint (one file per rank) for scripts/06_checkpoint_formats.py.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nanolm.distributed import configure_local_gloo
from nanolm.ddp_manual import shard_batch
from nanolm.fsdp_utils import apply_fsdp2, rank_state_bytes
from nanolm.model import GPTConfig, MiniGPT


def describe(tensor) -> str:
    kind = type(tensor).__name__
    local = tensor.to_local().shape if hasattr(tensor, "to_local") else tensor.shape
    return f"{kind:9s} local {tuple(local)} of {tuple(tensor.shape)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--keep-after-forward", action="store_true",
                        help="reshard_after_forward=False: keep gathered weights until backward")
    parser.add_argument("--save", help="directory for full and sharded checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(1)
    configure_local_gloo()
    if int(os.environ.get("WORLD_SIZE", 1)) < 2:
        raise SystemExit("Launch with torchrun --nproc_per_node=2 (or more).")
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    log = print if rank == 0 else (lambda *a, **k: None)
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        from torch.distributed.device_mesh import init_device_mesh

        cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
        torch.manual_seed(0)  # identical initialization before sharding
        model = MiniGPT(cfg)
        reference = MiniGPT(cfg)
        reference.load_state_dict(model.state_dict())
        replicated = {name: p.numel() * p.element_size() for name, p in reference.named_parameters()}
        full_param_bytes = sum(replicated.values())

        mesh = init_device_mesh("cpu", (world,))
        apply_fsdp2(model, mesh, reshard_after_forward=not args.keep_after_forward)
        log(f"FSDP2 on {world} ranks, reshard_after_forward={not args.keep_after_forward}")

        tracing = {"on": True}
        watched = "mlp.c_fc.weight"

        def before(module, _inputs):
            if tracing["on"]:
                log(f"  {module._trace_name} forward begins: {watched} is {describe(module.mlp.c_fc.weight)}")

        def after(module, _inputs, _output):
            if tracing["on"]:
                log(f"  {module._trace_name} forward ends:   {watched} is {describe(module.mlp.c_fc.weight)}")

        for i, block in enumerate(model.transformer.h):
            block._trace_name = f"block {i}"
            block.register_forward_pre_hook(before)
            block.register_forward_hook(after)

        hyper = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), **hyper)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), **hyper)

        data = torch.Generator().manual_seed(123)
        for step in range(1, args.steps + 1):
            tokens = torch.randint(0, cfg.vocab_size, (2 * world, 9), generator=data)
            local = shard_batch(tokens, rank, world)
            if step == 1:
                log("step 1 lifecycle (rank 0):")
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(local[:, :-1].contiguous(), targets=local[:, 1:].contiguous())
            loss.backward()
            if step == 1:
                log(f"  after backward: weight {describe(model.transformer.h[0].mlp.c_fc.weight)}, "
                    f"grad {describe(model.transformer.h[0].mlp.c_fc.weight.grad)}")
                tracing["on"] = False
            optimizer.step()

            ref_optimizer.zero_grad(set_to_none=True)
            _, ref_loss, _ = reference(tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
            ref_loss.backward()
            ref_optimizer.step()

            gathered = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
            error = max((gathered[k] - v).abs().max().item() for k, v in reference.state_dict().items() if k in gathered)
            log(f"step {step}: |param - reference| = {error:.2e}")
            ok = torch.tensor(float(error < 5e-5))
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() < 1:
                log("FAIL: FSDP2 update differs from the global-batch reference.")
                raise SystemExit(1)

        # Gradients are freed by zero_grad; measure them from one more backward.
        _, loss, _ = model(local[:, :-1].contiguous(), targets=local[:, 1:].contiguous())
        loss.backward()
        measured = rank_state_bytes(model, optimizer)
        replicated_state = {"parameters": full_param_bytes, "gradients": full_param_bytes,
                            "optimizer": 2 * full_param_bytes}
        log("\npersistent state on rank 0 (fp32, AdamW), measured with DTensor.to_local():")
        log(f"  {'':12s} {'replicated (DDP)':>18s} {'FSDP2 measured':>16s} {'ratio':>7s}")
        for key in ("parameters", "gradients", "optimizer"):
            log(f"  {key:12s} {replicated_state[key]:18,d} {measured[key]:16,d} {replicated_state[key] / measured[key]:7.2f}")
        log("  excluded: gathered unit weights, activations, communication buffers, allocator cache")

        if args.save:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            out = Path(args.save)
            full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if rank == 0:
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"model": full, "config": vars(cfg), "step": args.steps}, out / "full.pt")
            model_sd, optim_sd = get_state_dict(model, optimizer)
            dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(out / "sharded"))
            dist.barrier()
            log(f"\nsaved {out / 'full.pt'} (gathered on rank 0) and {out / 'sharded'}/ (one file per rank)")
        log(f"PASS: FSDP2 on {world} ranks matches one global-batch AdamW run.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
