"""Analytical Parallelism and Cluster Sizing Calculator.

Covers:
- Module 9: Context, pipeline, and expert parallelism (cost models, bubbles, memory & hierarchical communication calculator).
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
    n_experts: int = 1
    top_k: int = 1


@dataclass
class ParallelismConfig:
    dp_size: int = 1   # Replicated Data Parallel
    fsdp_size: int = 1 # Fully Sharded Data Parallel
    tp_size: int = 1   # Tensor Parallel
    pp_size: int = 1   # Pipeline Parallel
    cp_size: int = 1   # Context Parallel
    ep_size: int = 1   # Expert Parallel
    microbatches: int = 8


def compute_param_count(specs: ModelSpecs) -> int:
    """Return total parameter count for standard decoder-only transformer with MoE support."""
    embed = specs.vocab_size * specs.d_model
    per_layer_attn = 4 * (specs.d_model ** 2) + 2 * specs.d_model
    # If MoE, MLP parameters scale with n_experts
    mlp_experts = specs.n_experts if specs.n_experts > 1 else 1
    per_layer_mlp = mlp_experts * (8 * (specs.d_model ** 2)) + 2 * specs.d_model
    per_layer = per_layer_attn + per_layer_mlp
    return embed + specs.n_layer * per_layer


def compute_pipeline_bubble_fraction(pp_stages: int, microbatches: int) -> float:
    """Calculate 1F1B schedule pipeline bubble fraction: (p - 1) / (m + p - 1)."""
    if pp_stages <= 1:
        return 0.0
    return (pp_stages - 1) / (microbatches + pp_stages - 1)


def compute_communication_bytes_per_step(
    specs: ModelSpecs,
    parallelism: ParallelismConfig,
    batch_size_per_gpu: int = 2,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Calculate communicated bytes per step/layer across parallel axes.
    
    Order from outer to inner network domain:
    PP  -> DP -> FSDP -> EP -> TP
    """
    tokens_per_gpu = (specs.seq_len / parallelism.cp_size) * batch_size_per_gpu
    params = compute_param_count(specs)
    sharded_params = params / (parallelism.tp_size * parallelism.pp_size * parallelism.ep_size)

    # 1. PP: Point-to-point boundary activations per micro-batch
    # Volume: B_micro * (T / C) * d_model * dtype
    pp_bytes_per_microbatch = tokens_per_gpu * specs.d_model * dtype_bytes
    pp_total_bytes_step = pp_bytes_per_microbatch * parallelism.microbatches if parallelism.pp_size > 1 else 0

    # 2. DP: Gradient All-Reduce once per step
    # Volume: 2 * sharded_params * dtype
    dp_allreduce_bytes = 2 * sharded_params * dtype_bytes if parallelism.dp_size > 1 else 0

    # 3. FSDP: All-Gather weights (fwd & bwd) + Reduce-Scatter grads (bwd)
    # Volume: 3 * sharded_params * dtype per step
    fsdp_bytes_per_step = 3 * sharded_params * dtype_bytes if parallelism.fsdp_size > 1 else 0

    # 4. EP: All-to-All token dispatch and combine per MoE layer
    # Volume per layer: 2 * tokens_per_gpu * top_k * d_model * dtype (in fwd and bwd)
    ep_bytes_per_layer = 2 * (tokens_per_gpu * specs.top_k * specs.d_model * dtype_bytes) if parallelism.ep_size > 1 else 0
    ep_total_bytes_step = ep_bytes_per_layer * (specs.n_layer / parallelism.pp_size) * parallelism.microbatches

    # 5. TP: 2 All-Reduces per layer in fwd, 2 in bwd (blocking on critical path)
    # Volume per layer: 4 * tokens_per_gpu * d_model * dtype
    tp_bytes_per_layer = 4 * tokens_per_gpu * specs.d_model * dtype_bytes if parallelism.tp_size > 1 else 0
    tp_total_bytes_step = tp_bytes_per_layer * (specs.n_layer / parallelism.pp_size) * parallelism.microbatches

    mb = 1024 ** 2
    return {
        "pp_bytes_per_step_mb": round(pp_total_bytes_step / mb, 2),
        "dp_bytes_per_step_mb": round(dp_allreduce_bytes / mb, 2),
        "fsdp_bytes_per_step_mb": round(fsdp_bytes_per_step / mb, 2),
        "ep_bytes_per_step_mb": round(ep_total_bytes_step / mb, 2),
        "tp_bytes_per_step_mb": round(tp_total_bytes_step / mb, 2),
        "outer_to_inner_hierarchy": [
            ("PP", "Cross-Rack / Slow Links", "P2P boundary activations only", round(pp_total_bytes_step / mb, 2)),
            ("DP", "Inter-Node Spine-Leaf", "Gradient All-Reduce once per step (overlapped)", round(dp_allreduce_bytes / mb, 2)),
            ("FSDP", "Intra-Pod InfiniBand", "All-Gather + Reduce-Scatter (3x params per step)", round(fsdp_bytes_per_step / mb, 2)),
            ("EP", "Intra-Pod Fabric", "All-to-All token exchange per MoE layer", round(ep_total_bytes_step / mb, 2)),
            ("TP", "Intra-Node NVLink", "Blocking inner-loop All-Reduce (4x per layer)", round(tp_total_bytes_step / mb, 2)),
        ],
    }


def calculate_memory_and_scaling(
    specs: ModelSpecs,
    parallelism: ParallelismConfig,
    batch_size_per_gpu: int = 2,
    dtype_bytes: int = 2,
) -> Dict[str, Any]:
    """Calculate memory per GPU (GB), pipeline bubble efficiency, and communication profile."""
    total_gpus = (
        parallelism.dp_size
        * parallelism.fsdp_size
        * parallelism.tp_size
        * parallelism.pp_size
        * parallelism.cp_size
        * parallelism.ep_size
    )
    params = compute_param_count(specs)

    # Sharding across TP, PP, and EP
    sharded_params = params / (parallelism.tp_size * parallelism.pp_size * parallelism.ep_size)

    # Static memory per GPU (in Bytes):
    weight_bytes = sharded_params * dtype_bytes
    grad_bytes = sharded_params * dtype_bytes
    # ZeRO-1 / FSDP divides optimizer state across DP / FSDP
    effective_dp = max(1, parallelism.dp_size * parallelism.fsdp_size)
    optimizer_bytes = (sharded_params * 16) / effective_dp

    # Activation memory per layer
    tokens_per_gpu = (specs.seq_len / parallelism.cp_size) * batch_size_per_gpu
    layers_per_stage = specs.n_layer / parallelism.pp_size
    act_bytes_per_layer = tokens_per_gpu * specs.d_model * (10 + (24 / parallelism.tp_size)) * dtype_bytes
    total_act_bytes = layers_per_stage * act_bytes_per_layer

    gb = 1024 ** 3
    bubble_fraction = compute_pipeline_bubble_fraction(parallelism.pp_size, parallelism.microbatches)
    comm_profile = compute_communication_bytes_per_step(specs, parallelism, batch_size_per_gpu, dtype_bytes)

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
        "communication": comm_profile,
    }
