"""Analytical Parallelism and Cluster Sizing Calculator.

Covers:
- Session 10: Context and pipeline parallelism (cost models, bubbles, memory & communication calculator).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelSpecs:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    seq_len: int
    vocab_size: int = 50257


@dataclass
class ParallelismConfig:
    dp_size: int = 1   # Data Parallel
    tp_size: int = 1   # Tensor Parallel
    pp_size: int = 1   # Pipeline Parallel
    cp_size: int = 1   # Context Parallel
    microbatches: int = 8


def compute_param_count(specs: ModelSpecs) -> int:
    """Return total parameter count for standard decoder-only transformer."""
    # Embedding: vocab_size * d_model
    embed = specs.vocab_size * specs.d_model
    # Per layer:
    # Attn: 4 * d_model^2 (Q, K, V, Out)
    # MLP: 8 * d_model^2 (FC1, FC2)
    # LayerNorms: 4 * d_model
    per_layer = 12 * (specs.d_model ** 2) + 4 * specs.d_model
    return embed + specs.n_layer * per_layer


def compute_pipeline_bubble_fraction(pp_stages: int, microbatches: int) -> float:
    """Calculate 1F1B schedule pipeline bubble fraction: (p - 1) / (m + p - 1) or ~ (p - 1) / m."""
    if pp_stages <= 1:
        return 0.0
    return (pp_stages - 1) / (microbatches + pp_stages - 1)


def calculate_memory_and_scaling(
    specs: ModelSpecs,
    parallelism: ParallelismConfig,
    batch_size_per_gpu: int = 2,
    dtype_bytes: int = 2,  # bf16/fp16 = 2 bytes
) -> Dict[str, Any]:
    """Calculate memory per GPU (GB) and pipeline bubble efficiency."""
    total_gpus = parallelism.dp_size * parallelism.tp_size * parallelism.pp_size * parallelism.cp_size
    params = compute_param_count(specs)

    # Sharding across TP and PP
    sharded_params = params / (parallelism.tp_size * parallelism.pp_size)

    # Static memory per GPU (in Bytes):
    # Weights: 2 bytes
    weight_bytes = sharded_params * dtype_bytes
    # Gradients: 2 bytes
    grad_bytes = sharded_params * dtype_bytes
    # AdamW optimizer states (fp32 master weights + m + v = 16 bytes per param)
    # If ZeRO-1 / FSDP is applied along DP, optimizer states are divided by dp_size:
    optimizer_bytes = (sharded_params * 16) / parallelism.dp_size

    # Activation memory per layer (approx for selective checkpointing)
    # ~ (seq_len / cp_size) * (batch_size_per_gpu) * d_model * (10 + 24/tp_size) * bytes
    tokens_per_gpu = (specs.seq_len / parallelism.cp_size) * batch_size_per_gpu
    layers_per_stage = specs.n_layer / parallelism.pp_size
    act_bytes_per_layer = tokens_per_gpu * specs.d_model * (10 + (24 / parallelism.tp_size)) * dtype_bytes
    total_act_bytes = layers_per_stage * act_bytes_per_layer

    gb = 1024 ** 3
    bubble_fraction = compute_pipeline_bubble_fraction(parallelism.pp_size, parallelism.microbatches)

    return {
        "model_name": specs.name,
        "total_parameters_billions": round(params / 1e9, 3),
        "total_gpus": total_gpus,
        "pipeline_bubble_percent": round(bubble_fraction * 100, 2),
        "weights_per_gpu_gb": round(weight_bytes / gb, 2),
        "grads_per_gpu_gb": round(grad_bytes / gb, 2),
        "optimizer_per_gpu_gb": round(optimizer_bytes / gb, 2),
        "activations_per_gpu_gb": round(total_act_bytes / gb, 2),
        "estimated_peak_memory_per_gpu_gb": round(
            (weight_bytes + grad_bytes + optimizer_bytes + total_act_bytes) / gb, 2
        ),
    }
