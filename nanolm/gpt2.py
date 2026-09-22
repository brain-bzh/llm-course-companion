"""Utilities for validating NanoLM against an official GPT-2 checkpoint.

The conversion code is provided to students: Session 1 is about implementing
the Transformer, not reverse-engineering checkpoint names and tensor layouts.
"""

from collections.abc import Mapping
from typing import Any

import torch

from .model import GPTConfig, MiniGPT


def config_from_hf_gpt2(hf_config: Any) -> GPTConfig:
    """Translate a Hugging Face GPT-2 configuration into a NanoLM config."""
    return GPTConfig(
        vocab_size=hf_config.vocab_size,
        block_size=hf_config.n_positions,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_embd=hf_config.n_embd,
        dropout=0.0,
        bias=True,
        layer_norm_epsilon=hf_config.layer_norm_epsilon,
        use_sdpa=False,
    )


@torch.no_grad()
def copy_hf_gpt2_weights(model: MiniGPT, state: Mapping[str, torch.Tensor]) -> None:
    """Copy a Hugging Face GPT-2 state dict into an equivalent NanoLM model.

    Hugging Face stores Q, K and V in one ``Conv1D`` tensor whose matrix axes
    are the reverse of ``nn.Linear``. NanoLM keeps the three projections
    separate because that layout is easier to understand in Session 1.
    """

    def copy(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
        if target.shape != source.shape:
            raise ValueError(
                f"Shape mismatch for {name}: NanoLM {tuple(target.shape)} "
                f"!= GPT-2 {tuple(source.shape)}"
            )
        target.copy_(source)

    copy(model.transformer.wte.weight, state["transformer.wte.weight"], "wte.weight")
    copy(model.transformer.wpe.weight, state["transformer.wpe.weight"], "wpe.weight")

    width = model.config.n_embd
    for layer_index, block in enumerate(model.transformer.h):
        source_prefix = f"transformer.h.{layer_index}"

        copy(block.ln_1.weight, state[f"{source_prefix}.ln_1.weight"], f"h.{layer_index}.ln_1.weight")
        copy(block.ln_1.bias, state[f"{source_prefix}.ln_1.bias"], f"h.{layer_index}.ln_1.bias")

        fused_weight = state[f"{source_prefix}.attn.c_attn.weight"]
        fused_bias = state[f"{source_prefix}.attn.c_attn.bias"]
        q_weight, k_weight, v_weight = fused_weight.split(width, dim=1)
        q_bias, k_bias, v_bias = fused_bias.split(width, dim=0)
        copy(block.attn.w_q.weight, q_weight.T, f"h.{layer_index}.attn.w_q.weight")
        copy(block.attn.w_k.weight, k_weight.T, f"h.{layer_index}.attn.w_k.weight")
        copy(block.attn.w_v.weight, v_weight.T, f"h.{layer_index}.attn.w_v.weight")
        copy(block.attn.w_q.bias, q_bias, f"h.{layer_index}.attn.w_q.bias")
        copy(block.attn.w_k.bias, k_bias, f"h.{layer_index}.attn.w_k.bias")
        copy(block.attn.w_v.bias, v_bias, f"h.{layer_index}.attn.w_v.bias")
        copy(
            block.attn.c_proj.weight,
            state[f"{source_prefix}.attn.c_proj.weight"].T,
            f"h.{layer_index}.attn.c_proj.weight",
        )
        copy(
            block.attn.c_proj.bias,
            state[f"{source_prefix}.attn.c_proj.bias"],
            f"h.{layer_index}.attn.c_proj.bias",
        )

        copy(block.ln_2.weight, state[f"{source_prefix}.ln_2.weight"], f"h.{layer_index}.ln_2.weight")
        copy(block.ln_2.bias, state[f"{source_prefix}.ln_2.bias"], f"h.{layer_index}.ln_2.bias")
        copy(
            block.mlp.c_fc.weight,
            state[f"{source_prefix}.mlp.c_fc.weight"].T,
            f"h.{layer_index}.mlp.c_fc.weight",
        )
        copy(block.mlp.c_fc.bias, state[f"{source_prefix}.mlp.c_fc.bias"], f"h.{layer_index}.mlp.c_fc.bias")
        copy(
            block.mlp.c_proj.weight,
            state[f"{source_prefix}.mlp.c_proj.weight"].T,
            f"h.{layer_index}.mlp.c_proj.weight",
        )
        copy(
            block.mlp.c_proj.bias,
            state[f"{source_prefix}.mlp.c_proj.bias"],
            f"h.{layer_index}.mlp.c_proj.bias",
        )

    copy(model.transformer.ln_f.weight, state["transformer.ln_f.weight"], "ln_f.weight")
    copy(model.transformer.ln_f.bias, state["transformer.ln_f.bias"], "ln_f.bias")


def load_pretrained_gpt2(model_name: str = "gpt2") -> tuple[MiniGPT, torch.nn.Module]:
    """Load an official Hugging Face checkpoint into NanoLM.

    Returns both models so the caller can verify logit parity. ``transformers``
    is an optional dependency installed with ``uv sync --extra gpt2``.
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as error:
        raise RuntimeError(
            "GPT-2 validation requires the optional dependency: "
            "run `uv sync --extra gpt2`."
        ) from error

    reference = GPT2LMHeadModel.from_pretrained(model_name)
    reference.eval()

    model = MiniGPT(config_from_hf_gpt2(reference.config))
    copy_hf_gpt2_weights(model, reference.state_dict())
    model.eval()
    return model, reference
