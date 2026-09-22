"""Progressive tests for Module 1 — Transformer from first principles."""

import torch

from nanolm.gpt2 import copy_hf_gpt2_weights
from nanolm.model import (
    CausalSelfAttention,
    GPTConfig,
    MiniGPT,
    TransformerBlock,
)


def small_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        use_sdpa=False,
    )


def test_attention_output_shape():
    config = small_config()
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 8, config.n_embd)

    assert attention(x).shape == x.shape


def test_attention_is_causal():
    """Changing future inputs must not change earlier attention outputs."""
    torch.manual_seed(42)
    attention = CausalSelfAttention(small_config())
    attention.eval()

    x1 = torch.randn(1, 6, 32)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn(1, 2, 32)

    with torch.no_grad():
        y1 = attention(x1)
        y2 = attention(x2)

    torch.testing.assert_close(y1[:, :4], y2[:, :4], rtol=0.0, atol=1e-6)


def test_transformer_block_output_shape():
    config = small_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 8, config.n_embd)

    assert block(x).shape == x.shape


def test_model_output_shape_and_weight_tying():
    config = small_config()
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss, cache = model(x)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None
    assert cache is None
    assert model.transformer.wte.weight is model.lm_head.weight


def test_model_is_causal():
    """Changing future tokens must not change earlier next-token logits."""
    model = MiniGPT(small_config())
    model.eval()
    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    seq2 = torch.tensor([[10, 20, 30, 7, 8]])

    with torch.no_grad():
        logits1, _, _ = model(seq1)
        logits2, _, _ = model(seq2)

    torch.testing.assert_close(logits1[:, :3], logits2[:, :3], rtol=0.0, atol=1e-5)


def test_gpt2_weight_mapping_contract():
    """The provided converter must map fused GPT-2 tensors into student modules."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        bias=True,
    )
    model = MiniGPT(config)
    width = config.n_embd
    prefix = "transformer.h.0"
    state = {
        "transformer.wte.weight": torch.randn(config.vocab_size, width),
        "transformer.wpe.weight": torch.randn(config.block_size, width),
        f"{prefix}.ln_1.weight": torch.randn(width),
        f"{prefix}.ln_1.bias": torch.randn(width),
        f"{prefix}.attn.c_attn.weight": torch.randn(width, 3 * width),
        f"{prefix}.attn.c_attn.bias": torch.randn(3 * width),
        f"{prefix}.attn.c_proj.weight": torch.randn(width, width),
        f"{prefix}.attn.c_proj.bias": torch.randn(width),
        f"{prefix}.ln_2.weight": torch.randn(width),
        f"{prefix}.ln_2.bias": torch.randn(width),
        f"{prefix}.mlp.c_fc.weight": torch.randn(width, 4 * width),
        f"{prefix}.mlp.c_fc.bias": torch.randn(4 * width),
        f"{prefix}.mlp.c_proj.weight": torch.randn(4 * width, width),
        f"{prefix}.mlp.c_proj.bias": torch.randn(width),
        "transformer.ln_f.weight": torch.randn(width),
        "transformer.ln_f.bias": torch.randn(width),
    }

    copy_hf_gpt2_weights(model, state)

    fused_q, fused_k, fused_v = state[f"{prefix}.attn.c_attn.weight"].split(width, dim=1)
    torch.testing.assert_close(model.transformer.h[0].attn.w_q.weight, fused_q.T)
    torch.testing.assert_close(model.transformer.h[0].attn.w_k.weight, fused_k.T)
    torch.testing.assert_close(model.transformer.h[0].attn.w_v.weight, fused_v.T)
    torch.testing.assert_close(model.transformer.h[0].mlp.c_fc.weight, state[f"{prefix}.mlp.c_fc.weight"].T)
    assert model.transformer.wte.weight is model.lm_head.weight
