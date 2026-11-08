"""Session 12 starter — FLOPs, MFU and static memory accounting.

Copy this file to `nanolm/profile_utils.py` on your lab branch, implement the
three functions from the Module 4 derivations, then run:

    uv run pytest tests/test_module_04_profile.py -q

These numbers are predictions. The benchmark script turns them into a
measured comparison: tokens/s, peak memory, and (on a GPU) MFU.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .model import MiniGPT, GPTConfig


def compute_flops_per_token(config: GPTConfig) -> int:
    """Calculate theoretical forward + backward FLOPs per token for a decoder-only Transformer.

    Approximate formula:
    - Linear projections (Q, K, V, Out, MLP FC1, MLP FC2): 2 * N_params_non_embed
    - Attention matrix products (QK^T and Attn*V): 4 * n_layer * n_head * head_dim * seq_len
    - Backward pass roughly doubles forward pass (total ~ 3 * 2 * N = 6 * N + attention overhead).
    """
    # Matrix multiplies only: block projections + dense vocabulary head.
    # Embedding lookup is not a V*d matmul, but the tied output head is.
    # TODO: count matrix-multiply parameters: 12 * n_layer * n_embd^2 for the blocks
    #       (Q, K, V, output, and the two 4x MLP projections) plus vocab_size * n_embd
    #       for the tied output head (the embedding lookup is not a matmul).
    # TODO: training FLOPs = 6 * matrix_params (2 forward + 4 backward per weight)
    #       + attention scores/values: 12 * n_layer * n_embd * block_size.
    raise NotImplementedError


def estimate_mfu(
    tokens_per_sec: float,
    flops_per_token: int,
    gpu_peak_tflops: float,
) -> float:
    """Compute Model FLOPs Utilization (MFU).

    MFU = (measured_flops_per_second) / (gpu_peak_flops).
    gpu_peak_tflops is the theoretical peak of the GPU in TFLOP/s (e.g., A100 BF16 is ~312 TFLOPS).
    """
    # TODO: reject a non-positive peak with ValueError.
    # TODO: achieved TFLOP/s = tokens_per_sec * flops_per_token / 1e12; return it as a
    #       percentage of gpu_peak_tflops.
    raise NotImplementedError


def memory_breakdown(model: MiniGPT, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Calculate theoretical memory footprint (in Megabytes) for weights, grads, and AdamW states."""
    # TODO: bytes of parameters (numel * element_size), gradients (trainable params),
    #       and AdamW moments (two fp32 tensors per trainable parameter = 8 bytes each).
    # TODO: return megabytes (bytes / 2**20) under the keys parameters_mb, gradients_mb,
    #       optimizer_states_mb and total_static_mb.
    raise NotImplementedError
