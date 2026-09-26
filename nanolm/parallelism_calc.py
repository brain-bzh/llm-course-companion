"""Module 8: explicit, idealized device and persistent-state accounting.

D counts independent micro-batches. F and EP subdivide D; neither adds devices.
This model supports dense FSDP or unsharded MoE, not combined EP/FSDP layouts.
It does not predict peak VRAM, network time, or an optimal placement.
"""
from dataclasses import dataclass


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
    dp_size: int = 1  # Total data-parallel degree D, including sharded ranks.
    fsdp_size: int = 1  # F divides D; remaining D/F groups replicate shards.
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1  # EP divides D; experts have D/EP replicas.
    microbatches: int = 8


def compute_param_parts(specs: ModelSpecs) -> tuple[int, int]:
    """Bias/norm/router-free estimate, tied embedding and 4d GELU MLPs."""
    shared = specs.vocab_size * specs.d_model + specs.n_layer * 4 * specs.d_model**2
    experts = specs.n_layer * specs.n_experts * 8 * specs.d_model**2
    return shared, experts


def compute_param_count(specs: ModelSpecs) -> int:
    return sum(compute_param_parts(specs))


def compute_pipeline_bubble_fraction(pp_stages: int, microbatches: int) -> float:
    """Ideal balanced AFAB/non-interleaved 1F1B; neglect communication."""
    if pp_stages < 1 or microbatches < 1:
        raise ValueError("Stages and microbatches must be positive")
    return (pp_stages - 1) / (microbatches + pp_stages - 1)


def _validate(specs, cfg, batch_size, dtype_bytes):
    values = [specs.n_layer, specs.n_head, specs.d_model, specs.seq_len,
              specs.vocab_size, specs.n_experts, specs.top_k,
              *vars(cfg).values(), batch_size, dtype_bytes]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError("Dimensions, degrees, and byte counts must be positive integers")
    if specs.d_model % specs.n_head or specs.n_head % cfg.tp_size:
        raise ValueError("This MHA model requires d divisible by heads and heads by TP")
    if specs.n_layer % cfg.pp_size or specs.seq_len % cfg.cp_size:
        raise ValueError("This balanced model requires layers/PP and sequence/CP to be integral")
    if cfg.dp_size % cfg.fsdp_size or cfg.dp_size % cfg.ep_size:
        raise ValueError("FSDP and EP must divide the total data-parallel degree")
    if specs.n_experts % cfg.ep_size or specs.top_k > specs.n_experts:
        raise ValueError("EP must divide experts and top-k cannot exceed expert count")
    if cfg.fsdp_size > 1 and cfg.ep_size > 1:
        raise ValueError("Combined EP/FSDP needs separate expert shard groups; not modeled")


def calculate_memory_and_scaling(specs, parallelism, batch_size_per_gpu=2, dtype_bytes=2):
    """Return ideal balanced persistent state, excluding activations/transient buffers.

    AdamW assumption: fp32 gradients and two fp32 moments, plus fp32 master
    weights for a reduced-precision parameter copy. Embeddings are averaged over
    pipeline stages, so the largest stage may exceed this average substantially.
    """
    c = parallelism
    _validate(specs, c, batch_size_per_gpu, dtype_bytes)
    shared, expert = compute_param_parts(specs)
    local_params = (shared + expert / c.ep_size) / (c.tp_size * c.pp_size)
    resident = local_params / c.fsdp_size
    weights = resident * dtype_bytes
    grads = resident * 4
    optimizer = resident * (8 + (4 if dtype_bytes < 4 else 0))
    return {
        "model_name": specs.name,
        "total_parameters": shared + expert,
        "total_gpus": c.dp_size * c.tp_size * c.pp_size * c.cp_size,
        "global_tokens_per_step": batch_size_per_gpu * specs.seq_len * c.microbatches * c.dp_size,
        "local_parameters_before_fsdp": local_params,
        "pipeline_bubble_fraction": compute_pipeline_bubble_fraction(c.pp_size, c.microbatches),
        "weights_bytes": weights,
        "gradients_bytes": grads,
        "optimizer_bytes": optimizer,
        "persistent_state_bytes": weights + grads + optimizer,
        "exclusions": "Activations, gathered weights, communication buffers, allocator reserve, stage imbalance",
    }


def ring_allreduce_sent_bytes(payload_bytes: int, ranks: int) -> float:
    """Bytes sent per rank by an ideal ring; excludes received bytes and latency."""
    if payload_bytes < 0 or ranks < 1:
        raise ValueError("Payload must be nonnegative and ranks positive")
    return 2 * (ranks - 1) / ranks * payload_bytes
