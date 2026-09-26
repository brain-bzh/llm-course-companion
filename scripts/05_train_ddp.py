"""Module 5: DDP accumulation compared with the identical global batch.

Run: uv run torchrun --standalone --nproc_per_node=2 scripts/05_train_ddp.py
This is a correctness experiment on synthetic data, not a scaling benchmark.
"""
import argparse
from contextlib import nullcontext
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nanolm.model import MiniGPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--accum', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if min(args.accum, args.steps) < 1:
        parser.error('accum and steps must be positive')
    torch.set_num_threads(1)
    if sys.platform == 'darwin':
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        torch.manual_seed(42)
        cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32)
        raw = MiniGPT(cfg).to(device)
        ref = MiniGPT(cfg).to(device)
        ref.load_state_dict(raw.state_dict())
        model = DDP(raw, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else raw
        # SGD makes update equivalence easy to interpret without optimizer-state effects.
        opt, ref_opt = torch.optim.SGD(model.parameters(), lr=.05), torch.optim.SGD(ref.parameters(), lr=.05)
        batch, length = 2, 8
        generator = torch.Generator().manual_seed(123)
        for step in range(args.steps):
            tokens = torch.randint(0, 64, (world*args.accum*batch, length+1), generator=generator).to(device)
            own = tokens[rank*args.accum*batch:(rank+1)*args.accum*batch]
            opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for micro in range(args.accum):
                data = own[micro*batch:(micro+1)*batch]
                # Both forward and backward must be inside no_sync.
                context = model.no_sync() if world > 1 and micro+1 < args.accum else nullcontext()
                with context:
                    _, loss, _ = model(data[:, :-1].contiguous(), targets=data[:, 1:].contiguous())
                    (loss/args.accum).backward()
                loss_sum += loss.detach()*batch*length
            ref_opt.zero_grad(set_to_none=True)
            _, ref_loss, _ = ref(tokens[:, :-1].contiguous(), targets=tokens[:, 1:].contiguous())
            ref_loss.backward()
            for actual, expected in zip(raw.parameters(), ref.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
            opt.step()
            ref_opt.step()
            for actual, expected in zip(raw.parameters(), ref.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            count = torch.tensor(own.size(0)*length, device=device)
            if world > 1:
                dist.all_reduce(count)
                dist.all_reduce(loss_sum)
            assert count.item() == tokens.size(0)*length
            torch.testing.assert_close(loss_sum/count, ref_loss.detach(), atol=2e-6, rtol=2e-5)
            if rank == 0:
                print(f'step={step+1} global_tokens={count.item()} loss={(loss_sum/count).item():.6f}')
        if rank == 0:
            print(f'PASS: {world} ranks, accumulation={args.accum}; gradients, updates and token counts match.')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
