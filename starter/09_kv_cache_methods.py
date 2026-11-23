"""Session 26 scaffold — the KV-cache path, to paste into YOUR nanolm/model.py.

Unlike the other starters, this file does not replace a module: the model you
wrote in Module 1 travels with you. Each class below lists only the methods to
add or replace in the class of the same name in your model. Paste them, keep
your own __init__, then fill the four TODOs:

    1. concatenate the past keys and values;
    2. offset the causal mask by past_len;
    3. take the SDPA branch only without a cache;
    4. offset the learned positions by past_len.

Attribute names follow the reference model (w_q, n_head, head_dim, bias_mask,
attn_dropout, resid_dropout, config). Rename them to match yours. Your model
already imports math, torch, nn, F and typing.Optional; add List and Tuple.

    uv run pytest tests/test_module_09_kv_cache.py -k "every_position or chunks" -q

Guide: https://brain-bzh.github.io/llm-course/companion/09-kv-cache/
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The Module 1 contract, now a thin wrapper that discards the cache."""
        y, _ = self.forward_with_cache(x)
        return y

    def forward_with_cache(
        self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()
        q = self.w_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # TODO 1: prepend the cached keys and values along the sequence axis (dim=2).
            raise NotImplementedError

        new_cache = (k, v) if use_cache else None
        total_T = k.size(2)  # tokens seen so far, including the T new ones

        # TODO 3: is_causal=True assumes queries and keys start at the same position.
        #         When is that false? Add the missing condition.
        if self.config.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.config.dropout if self.training else 0.0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # B, H, T, total_T
            # TODO 2: the T new queries sit at absolute positions past_len .. past_len + T - 1.
            #         Compute past_len and slice the rows of the mask accordingly
            #         (the columns are :total_T). With one new token the row is all ones,
            #         so only a chunked test can catch a wrong offset.
            mask = self.bias_mask[:, :, :T, :total_T]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), new_cache


class TransformerBlock(nn.Module):
    def forward_with_cache(
        self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """Same two residual branches as forward(); the cache only passes through attention."""
        attn_out, new_cache = self.attn.forward_with_cache(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ):
        B, T = idx.size()
        past_len = 0 if kv_caches is None else kv_caches[0][0].size(2)
        assert past_len + T <= self.config.block_size, "sequence longer than block_size"

        # TODO 4: positions must continue from the cache, not restart at zero.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_cache = None if kv_caches is None else kv_caches[i]
            x, cache = block.forward_with_cache(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(cache)

        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss, new_caches
