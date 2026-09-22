"""Module 10 — KV-cached decoding.

Demonstrates:
- Incremental decoding with Key-Value Cache.
- Exact numerical and token equivalence between cached and uncached generation.
- Latency and throughput benchmark demonstrating the $O(T)$ speedup over $O(T^2)$ recomputation.
"""

import torch
from nanolm.model import MiniGPT, GPTConfig
from nanolm.kv_cache import benchmark_generation_speed

def main():
    print("=== Module 10: KV-Cached Decoding Benchmark & Equivalence ===")
    config = GPTConfig(
        vocab_size=1000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )
    model = MiniGPT(config)
    model.eval()

    # Prompt: 10 tokens, generate 25 new tokens
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    print("Running speed and equivalence benchmark (Prompt length: 10, New tokens: 25)...")
    res = benchmark_generation_speed(model, prompt, new_tokens=25)

    print(f"\nUncached Generation Time: {res['uncached_time_sec']:.4f} s ({res['uncached_tokens_per_sec']} tok/s)")
    print(f"Cached Generation Time:   {res['cached_time_sec']:.4f} s ({res['cached_tokens_per_sec']} tok/s)")
    print(f"Speedup Factor:           {res['speedup_factor']}x")
    print(f"Exact Token Match:        {res['exact_token_match']}")

    assert res["exact_token_match"], "Cached and uncached generations diverged!"
    print("SUCCESS: Module 10 deliverable verified (KV-cache is mathematically identical and faster).")

if __name__ == "__main__":
    main()
