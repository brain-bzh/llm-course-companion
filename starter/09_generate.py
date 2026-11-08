"""Session 26 starter — greedy/sampled generation with and without a KV cache.

Copy this file to `nanolm/generate.py` on your lab branch. `generate_uncached`
is provided as the reference behavior; implement `generate_cached`, then run:

    uv run pytest tests/test_module_09_kv_cache.py -q
    uv run python scripts/09_kv_cache_bench.py

Your model must accept `model(idx, kv_caches=..., use_cache=True)` and return
`(logits, loss, new_caches)`; see the Module 9 lab guide for the attention change.
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
    if max_new_tokens < 0 or idx.size(1) == 0:
        raise ValueError("Use a nonempty prompt and a nonnegative generation length")
    if max_new_tokens == 0:
        return idx
    if idx.size(1) + max_new_tokens - 1 > model.config.block_size:
        raise ValueError("Cached generation exceeds the learned-position context window")
    model.eval()
    # TODO 1 (prefill): run the WHOLE prompt once with use_cache=True and keep
    #         the returned per-layer (K, V) caches.
    # TODO 2 (loop, max_new_tokens times): select the next token from the logits
    #         of the LAST position, exactly as generate_uncached does (temperature,
    #         top-k, argmax when temperature == 0), and append it to the output.
    # TODO 3 (decode): unless this was the final token, run ONLY the new token
    #         (shape B x 1) with kv_caches=caches, use_cache=True.
    # Question: why does the final emitted token never need a forward pass?
    raise NotImplementedError
