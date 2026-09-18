"""Profiling utilities: FLOPs counter, MFU estimation, and memory breakdown.

Covers:
- Session 6: Single-GPU performance (FLOPs, throughput, MFU, memory accounting).
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
    N = (
        12 * config.n_layer * (config.n_embd ** 2)
        + 13 * config.n_layer * config.n_embd
    )
    # Forward + backward FLOPs per token
    flops_per_token = 6 * N + 12 * config.n_layer * config.n_head * (config.n_embd // config.n_head) * config.block_size
    return flops_per_token


def estimate_mfu(
    tokens_per_sec: float,
    flops_per_token: int,
    gpu_peak_tflops: float,
) -> float:
    """Compute Model FLOPs Utilization (MFU).

    MFU = (measured_flops_per_second) / (gpu_peak_flops).
    gpu_peak_tflops is the theoretical peak of the GPU in TFLOP/s (e.g., A100 BF16 is ~312 TFLOPS).
    """
    measured_tflops = (tokens_per_sec * flops_per_token) / 1e12
    return (measured_tflops / gpu_peak_tflops) * 100.0


def memory_breakdown(model: MiniGPT, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Calculate theoretical memory footprint (in Megabytes) for weights, grads, and AdamW states."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    # AdamW stores 2 states (first moment m and second moment v), typically in float32 (4 bytes each)
    adam_states_bytes = sum(p.numel() * 8 for p in model.parameters() if p.requires_grad)

    to_mb = 1.0 / (1024 * 1024)
    return {
        "parameters_mb": param_bytes * to_mb,
        "gradients_mb": grad_bytes * to_mb,
        "optimizer_states_mb": adam_states_bytes * to_mb,
        "total_static_mb": (param_bytes + grad_bytes + adam_states_bytes) * to_mb,
    }
