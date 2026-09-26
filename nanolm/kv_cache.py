"""Key-Value (KV) Cache structure and decoding analysis.

Covers:
- Module 11: KV-cached decoding (cache structure, prefill/decode timing, memory footprint).
"""

from typing import List, Tuple, Optional
import time
import torch
import torch.nn as nn
from .generate import generate_cached, generate_uncached


class KVCache:
    """Explicit KV-cache manager for Transformer inference."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key and value slices to layer cache."""
        if self.caches[layer_idx] is None:
            self.caches[layer_idx] = (k, v)
        else:
            past_k, past_v = self.caches[layer_idx]
            self.caches[layer_idx] = (
                torch.cat([past_k, k], dim=2),
                torch.cat([past_v, v], dim=2),
            )
        return self.caches[layer_idx]

    def reset(self):
        self.caches = [None] * self.n_layers

    @property
    def current_seq_len(self) -> int:
        if self.caches[0] is None:
            return 0
        return self.caches[0][0].size(2)


def compute_kv_cache_size_bytes(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # 2 for float16/bfloat16
) -> int:
    """Calculate KV cache memory footprint in bytes.

    Formula: 2 (K and V) * n_layers * batch_size * n_kv_heads * seq_len * head_dim * bytes_per_element.
    Shows the dramatic memory saving of GQA / MQA over MHA.
    """
    return 2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim * dtype_bytes


@torch.no_grad()
def benchmark_generation_speed(model, prompt_tokens, new_tokens=30, repeats=5):
    """Synchronised, warmed-up latency measurements on a fixed greedy workload.

    TTFT here is model prefill + token selection, excluding queue/network/tokenizer.
    Decode intervals include Python/selection overhead and exclude the first token.
    """
    import statistics
    if new_tokens < 2 or repeats < 1:
        raise ValueError("Use at least two output tokens and one measurement repeat")
    if prompt_tokens.size(1) + new_tokens - 1 > model.config.block_size:
        raise ValueError("Workload exceeds the fixed learned-position context window")
    device = prompt_tokens.device
    model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def cached_run():
        sync()
        begin = time.perf_counter()
        logits, _, caches = model(prompt_tokens, use_cache=True)
        token = logits[:, -1].argmax(-1, keepdim=True)
        sync()
        prefill = time.perf_counter() - begin
        output = [prompt_tokens, token]
        intervals = []
        for _ in range(new_tokens - 1):
            begin = time.perf_counter()
            logits, _, caches = model(token, kv_caches=caches, use_cache=True)
            token = logits[:, -1].argmax(-1, keepdim=True)
            sync()
            intervals.append(time.perf_counter() - begin)
            output.append(token)
        cache_bytes = sum(t.numel() * t.element_size() for pair in caches for t in pair)
        return torch.cat(output, dim=1), prefill, intervals, cache_bytes

    cached_run()
    generate_uncached(model, prompt_tokens, new_tokens, temperature=0)
    cached_times, uncached_times, prefills, decodes = [], [], [], []
    for _ in range(repeats):
        sync()
        begin = time.perf_counter()
        uncached = generate_uncached(model, prompt_tokens, new_tokens, temperature=0)
        sync()
        uncached_times.append(time.perf_counter() - begin)
        cached, prefill, intervals, cache_bytes = cached_run()
        if not torch.equal(cached, uncached):
            raise AssertionError("Greedy tokens differ; investigate logits before timing conclusions")
        cached_times.append(prefill + sum(intervals))
        prefills.append(prefill)
        decodes.extend(intervals)
    tc, tu = statistics.median(cached_times), statistics.median(uncached_times)
    return {
        "device": str(device), "batch_size": prompt_tokens.size(0),
        "prompt_tokens": prompt_tokens.size(1), "new_tokens": new_tokens, "repeats": repeats,
        "uncached_time_sec": tu, "cached_time_sec": tc,
        "uncached_tokens_per_sec": prompt_tokens.size(0) * new_tokens / tu,
        "cached_tokens_per_sec": prompt_tokens.size(0) * new_tokens / tc,
        "speedup_factor": tu / tc, "exact_token_match": True,
        "model_ttft_ms": 1000 * statistics.median(prefills),
        "decode_interval_median_ms": 1000 * statistics.median(decodes),
        "decode_interval_min_ms": 1000 * min(decodes),
        "decode_interval_max_ms": 1000 * max(decodes),
        "logical_cache_bytes": cache_bytes,
    }
