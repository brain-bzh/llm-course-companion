"""Autoregressive text generation utilities.

Covers:
- Module 4: Train a small GPT (sampling as a qualitative diagnostic).
- Module 11: KV-cached decoding (cached vs uncached generation).
"""

from typing import Optional, List
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_uncached(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively by recomputing the entire sequence forward pass at each step."""
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context to block_size if needed
        block_size = getattr(model.config, "block_size", 1024)
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        logits, _, _ = model(idx_cond)
        # Pluck logits at final step
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


@torch.no_grad()
def generate_cached(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens using Key-Value caching to avoid redundant attention computation."""
    model.eval()
    # 1. Prefill stage: process prompt
    logits, _, kv_caches = model(idx, use_cache=True)
    curr_idx = idx

    for _ in range(max_new_tokens):
        # Pluck logits at final step
        step_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
            step_logits[step_logits < v[:, [-1]]] = -float("Inf")

        if temperature > 0:
            probs = F.softmax(step_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(step_logits, dim=-1, keepdim=True)

        curr_idx = torch.cat((curr_idx, idx_next), dim=1)

        # 2. Decode stage: feed only the single new token
        logits, _, kv_caches = model(idx_next, kv_caches=kv_caches, use_cache=True)

    return curr_idx
