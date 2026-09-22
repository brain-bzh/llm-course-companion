"""Session 1 starter — build a decoder-only Transformer from first principles.

Copy this file over ``nanolm/model.py`` before starting the exercise.  Keep the
public class names and method signatures intact: the tests use them as the
exercise contract.

Guide: https://brain-bzh.github.io/llm-course/companion/01-transformer/
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    layer_norm_epsilon: float = 1e-5
    # Later sessions may replace explicit attention with PyTorch SDPA.
    use_sdpa: bool = False


class CausalSelfAttention(nn.Module):
    """Explicit multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # TODO: store the head dimensions.
        # TODO: define projections named w_q, w_k, w_v and c_proj.
        # TODO: register a lower-triangular causal mask as a buffer.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, T, d_model]`` to ``[B, T, d_model]``."""
        # TODO: project Q, K and V.
        # TODO: reshape them to [B, n_head, T, d_head].
        # TODO: compute scaled scores and mask future positions.
        # TODO: apply softmax, mix values, join heads and project.
        raise NotImplementedError


class MLP(nn.Module):
    """Per-token feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: define c_fc and c_proj for d_model -> 4*d_model -> d_model.
        # TODO: use GELU(approximate="tanh") and dropout.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with two residual branches."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: create ln_1, attn, ln_2 and mlp. Pass layer_norm_epsilon and
        #       bias=config.bias to both LayerNorms (keep their scale weights).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x = x + attention(norm(x)).
        # TODO: x = x + mlp(norm(x)).
        raise NotImplementedError


class MiniGPT(nn.Module):
    """Small decoder-only language model assembled from Transformer blocks."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # TODO: create transformer as a ModuleDict containing wte, wpe, drop,
        #       h (the block stack) and ln_f.
        # TODO: create the vocabulary projection and tie its weight to the
        #       token embedding weight.
        raise NotImplementedError

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """Return ``(logits, loss, None)`` for compatibility with later labs."""
        # TODO: combine token and position embeddings.
        # TODO: apply every block, the final norm and the vocabulary head.
        # TODO: compute cross-entropy when shifted targets are supplied.
        raise NotImplementedError

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of trainable parameters."""
        # TODO: count parameters and optionally exclude position embeddings.
        raise NotImplementedError
