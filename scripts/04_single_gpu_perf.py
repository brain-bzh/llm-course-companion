"""Module 4: warmed-up fixed-shape training ablations with declared hardware.

CPU default validates mechanics; request CUDA explicitly for a GPU experiment.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
import torch
from nanolm.model import MiniGPT, GPTConfig
from nanolm.optim import configure_optimizers
from nanolm.profile_utils import compute_flops_per_token, estimate_mfu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--modes', nargs='+', choices=['manual', 'sdpa', 'bf16', 'compile'], default=['manual', 'sdpa'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--peak-tflops', type=float, help='Dense peak at measured precision; do not use sparse peak')
    parser.add_argument('--trace-dir', help='Optional Chrome traces; collected separately from timing')
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats, args.batch_size, args.block_size) < 1:
        parser.error('Workload and timing counts must be positive')
    if args.peak_tflops is not None and (args.peak_tflops <= 0 or args.device != 'cuda'):
        parser.error('MFU needs a positive declared CUDA device peak')
    if args.peak_tflops and len(args.modes) > 1 and 'bf16' in args.modes:
        parser.error('Use a separate BF16 invocation: the hardware peak depends on precision')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; CPU results cannot substitute for GPU measurements')
    if 'bf16' in args.modes and args.device == 'cuda' and not torch.cuda.is_bf16_supported():
        parser.error('This CUDA device does not support BF16')
    torch.set_num_threads(1)
    device = torch.device(args.device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    results = []
    for mode in args.modes:
        torch.manual_seed(42)
        cfg = GPTConfig(vocab_size=512, block_size=args.block_size, n_layer=2, n_head=4,
                        n_embd=128, use_sdpa=mode != 'manual')
        raw = MiniGPT(cfg).to(device)
        model = torch.compile(raw) if mode == 'compile' else raw
        optimizer = configure_optimizers(raw, device_type=device.type)
        tokens = torch.randint(0, cfg.vocab_size, (args.batch_size, cfg.block_size+1), device=device)
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        dtype = torch.bfloat16 if mode == 'bf16' else torch.float32

        def update():
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device.type, dtype=dtype) if mode == 'bf16' else nullcontext()
            with context:
                _, loss, _ = model(x, targets=y)
            loss.backward()
            optimizer.step()

        for _ in range(args.warmup):
            update()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeats):
            sync()
            begin = time.perf_counter()
            for _ in range(args.steps):
                update()
            sync()
            samples.append(args.steps*args.batch_size*cfg.block_size/(time.perf_counter()-begin))
        median = statistics.median(samples)
        results.append({'mode': mode, 'tokens_per_sec_samples': samples, 'median_tokens_per_sec': median,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
                        'approx_mfu_percent': estimate_mfu(median, compute_flops_per_token(cfg), args.peak_tflops)
                        if args.peak_tflops else None})
        if args.trace_dir:
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as trace:
                update()
                sync()
            trace.export_chrome_trace(str(trace_dir / f'{mode}.json'))
        del model, raw, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps({'workload': vars(args), 'torch_version': torch.__version__,
                      'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU (not GPU MFU)',
                      'results': results}, indent=2))


if __name__ == '__main__':
    main()
