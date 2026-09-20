"""Minimal Decoder-Only Transformer (MiniGPT).

Covers:
- Module 1: Transformer from first principles (embeddings, causal attention, MLP, Pre-LN).
- Module 6: SDPA vs explicit attention toggle.
- Module 11: Key-Value cache support for incremental autoregressive decoding.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Standard GPT-2 vocabulary size
    block_size: int = 256    # Context window / max sequence length
    n_layer: int = 4         # Number of transformer layers
    n_head: int = 4          # Number of attention heads
    n_embd: int = 128        # Embedding dimension (d_model)
    dropout: float = 0.0     # Dropout probability
    bias: bool = False       # Use bias in Linears and LayerNorms (modern LLMs prefer False)
    use_sdpa: bool = True    # Use F.scaled_dot_product_attention (PyTorch Flash/efficient backend)


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with optional KV-cache support."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, query, value projections for all heads in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask for non-SDPA fallback and validation tests
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV-Cache handling (Module 11)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None
        total_T = k.size(2)

        if self.config.use_sdpa and hasattr(F, "scaled_dot_product_attention") and kv_cache is None:
            # Fast PyTorch SDPA (Module 6) for non-cached prefill/training
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Explicit attention computation (Module 1 & incremental decode in Module 11)
            # QK^T / sqrt(d_k) -> (B, n_head, T, total_T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if kv_cache is None:
                # Causal masking: mask future positions
                att = att.masked_fill(self.bias_mask[:, :, :T, :total_T] == 0, float("-inf"))
            # In incremental decode with past cache, each newly appended token can attend to all past tokens

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, head_dim)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache


class MLP(nn.Module):
    """Feed-Forward Network with Pre-LN Residual connection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN attention with residual
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        # Pre-LN MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    """Minimal Decoder-Only Generative Pretrained Transformer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = idx.device
        B, T = idx.size()

        if kv_caches is None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (T)
        else:
            # In incremental decode, the position of the new token is based on existing cached keys
            past_len = kv_caches[0][0].size(2)
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)     # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(updated_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, calculate cross-entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: forward the lm_head on the last position only if not using targets
            logits = self.lm_head(x)
            loss = None

        return logits, loss, new_caches

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
