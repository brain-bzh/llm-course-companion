"""Module 6 — Single-GPU performance.

Demonstrates:
- Memory breakdown (parameters, gradients, optimizer states).
- FLOPs calculation per token.
- MFU (Model FLOPs Utilization) estimation.
- Comparing PyTorch SDPA vs manual attention.
"""

import time
import torch
from minilm.model import MiniGPT, GPTConfig
from minilm.optim import configure_optimizers
from minilm.profile_utils import compute_flops_per_token, estimate_mfu, memory_breakdown

def main():
    print("=== Module 6: Single-GPU Performance & Profiling ===")
    config = GPTConfig(
        vocab_size=50257,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model)

    # 1. Theoretical Memory Accounting
    mem = memory_breakdown(model, optimizer)
    print("Static Memory Breakdown:")
    print(f"  Parameters:       {mem['parameters_mb']:.2f} MB")
    print(f"  Gradients:        {mem['gradients_mb']:.2f} MB")
    print(f"  AdamW States:     {mem['optimizer_states_mb']:.2f} MB")
    print(f"  Total Static VRAM:{mem['total_static_mb']:.2f} MB")

    # 2. FLOPs per token
    flops_per_tok = compute_flops_per_token(config)
    print(f"\nTheoretical FLOPs per token (Fwd+Bwd): {flops_per_tok:,} FLOPs")

    # 3. Micro-benchmark SDPA throughput
    x = torch.randint(0, config.vocab_size, (4, 256))
    y = torch.randint(0, config.vocab_size, (4, 256))

    # Warmup
    for _ in range(3):
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.zero_grad()

    steps = 10
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.perf_counter() - t0

    tokens_processed = steps * 4 * 256
    tok_per_sec = tokens_processed / elapsed
    print(f"\nMeasured Throughput: {tok_per_sec:.2f} tokens/s (over {steps} steps on CPU/MPS/CUDA)")

    # Example MFU estimation on hypothetical 100 TFLOPS accelerator
    mfu = estimate_mfu(tok_per_sec, flops_per_tok, gpu_peak_tflops=100.0)
    print(f"Estimated MFU (against 100 TFLOPS device): {mfu:.3f}%")

if __name__ == "__main__":
    main()
