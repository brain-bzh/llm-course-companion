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


def benchmark_generation_speed(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    new_tokens: int = 30,
) -> dict:
    """Measure speed and memory of uncached vs KV-cached generation."""
    # 1. Uncached run
    t0 = time.perf_counter()
    out_uncached = generate_uncached(model, prompt_tokens.clone(), max_new_tokens=new_tokens, temperature=0.0)
    t_uncached = time.perf_counter() - t0

    # 2. Cached run
    t1 = time.perf_counter()
    out_cached = generate_cached(model, prompt_tokens.clone(), max_new_tokens=new_tokens, temperature=0.0)
    t_cached = time.perf_counter() - t1

    tokens_per_sec_uncached = new_tokens / t_uncached if t_uncached > 0 else 0
    tokens_per_sec_cached = new_tokens / t_cached if t_cached > 0 else 0
    speedup = t_uncached / t_cached if t_cached > 0 else 1.0

    return {
        "uncached_time_sec": round(t_uncached, 4),
        "cached_time_sec": round(t_cached, 4),
        "uncached_tokens_per_sec": round(tokens_per_sec_uncached, 2),
        "cached_tokens_per_sec": round(tokens_per_sec_cached, 2),
        "speedup_factor": round(speedup, 2),
        "exact_token_match": torch.equal(out_uncached, out_cached),
    }
