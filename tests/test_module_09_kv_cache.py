"""Tests for Module 10 — KV-cached decoding."""

import torch
from nanolm.model import MiniGPT, GPTConfig
from nanolm.generate import generate_cached, generate_uncached


def test_kv_cached_vs_uncached_exact_match():
    """Verify that cached decoding produces the EXACT identical sequence of tokens as uncached generation."""
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = MiniGPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))

    # Deterministic greedy generation (temp=0.0)
    uncached_out = generate_uncached(model, prompt.clone(), max_new_tokens=15, temperature=0.0)
    cached_out = generate_cached(model, prompt.clone(), max_new_tokens=15, temperature=0.0)

    assert torch.equal(uncached_out, cached_out), "KV-cached output diverged from uncached output!"


def test_cached_logits_at_every_position_and_context_boundary():
    torch.manual_seed(123)
    for sdpa in (False, True):
        cfg = GPTConfig(vocab_size=32, block_size=12, n_layer=1, n_head=2,
                        n_embd=16, use_sdpa=sdpa)
        model = MiniGPT(cfg).eval()
        tokens = torch.randint(0, 32, (2, 12))
        caches = None
        with torch.no_grad():
            for i in range(12):
                full, _, _ = model(tokens[:, :i+1])
                cached, _, caches = model(tokens[:, i:i+1], kv_caches=caches, use_cache=True)
                torch.testing.assert_close(full[:, -1], cached[:, -1], atol=1e-5, rtol=1e-5)
        # Last emitted token needs no extra forward outside the position window.
        a = generate_cached(model, tokens[:, :10], 3, temperature=0)
        b = generate_uncached(model, tokens[:, :10], 3, temperature=0)
        assert torch.equal(a, b)
        assert torch.equal(generate_cached(model, tokens, 0), tokens)


def test_benchmark_reports_prefill_and_decode():
    from nanolm.kv_cache import benchmark_generation_speed
    cfg = GPTConfig(vocab_size=32, block_size=12, n_layer=1, n_head=2, n_embd=16)
    result = benchmark_generation_speed(MiniGPT(cfg), torch.ones(1, 4, dtype=torch.long), 3, repeats=2)
    assert result['model_ttft_ms'] > 0
    assert result['decode_interval_median_ms'] > 0
    # Prompt plus two decoded inputs; final output has not been forwarded.
    assert result['logical_cache_bytes'] == 2 * 1 * 1 * 2 * 6 * 8 * 4
