"""Module 9 — Context, pipeline, and expert parallelism.

Demonstrates:
- Analytical calculator for 5D parallelism (DP x FSDP x TP x PP x CP x EP).
- Pipeline bubble overhead calculation under 1F1B schedule.
- Per-GPU VRAM estimation across cluster topologies.
- Communication volume breakdown proving the outer-to-inner hierarchy:
  PP -> DP -> FSDP -> EP -> TP
"""

from nanolm.parallelism_calc import ModelSpecs, ParallelismConfig, calculate_memory_and_scaling


def main():
    print("=== Module 9: Multidimensional Parallelism & Cluster Sizing ===")

    # Case: 70B MoE model (e.g. 8 experts, top-2 active)
    moe70b = ModelSpecs(
        name="Mixtral/DeepSeek-Style-MoE",
        n_layer=32,
        n_head=32,
        d_model=4096,
        seq_len=4096,
        vocab_size=32000,
        n_experts=8,
        top_k=2,
    )

    scenarios = [
        ("Single Node (8 GPUs) - TP=8 (Intra-Node NVLink)", ParallelismConfig(dp_size=1, tp_size=8, pp_size=1, cp_size=1)),
        ("Cluster (32 GPUs) - DP=4, TP=4, PP=2 (Inter-Node)", ParallelismConfig(dp_size=4, tp_size=4, pp_size=2, cp_size=1, microbatches=8)),
        ("MoE Cluster (64 GPUs) - DP=2, TP=4, PP=2, EP=4", ParallelismConfig(dp_size=2, tp_size=4, pp_size=2, cp_size=1, ep_size=4, microbatches=16)),
        ("Long Context (128 GPUs) - DP=2, TP=4, PP=2, CP=4, EP=2", ParallelismConfig(dp_size=2, tp_size=4, pp_size=2, cp_size=4, ep_size=2, microbatches=16)),
    ]

    for label, cfg in scenarios:
        res = calculate_memory_and_scaling(moe70b, cfg, batch_size_per_gpu=1)
        print(f"\nScenario: {label}")
        print(f"  Total GPUs:                  {res['total_gpus']}")
        print(f"  Pipeline Bubble:             {res['pipeline_bubble_percent']}%")
        print(f"  Weights per GPU:             {res['weights_per_gpu_gb']} GB")
        print(f"  Gradients per GPU:           {res['grads_per_gpu_gb']} GB")
        print(f"  Optimizer per GPU (ZeRO-1):  {res['optimizer_per_gpu_gb']} GB")
        print(f"  Activations per GPU:         {res['activations_per_gpu_gb']} GB")
        print(f"  -> Estimated Peak VRAM:      {res['estimated_peak_memory_per_gpu_gb']} GB")

    # Deep dive into the outer-to-inner communication hierarchy:
    print("\n" + "=" * 70)
    print("THE OUTER-TO-INNER COMMUNICATION HIERARCHY (PP -> DP -> FSDP -> EP -> TP)")
    print("=" * 70)
    full_cfg = ParallelismConfig(dp_size=2, fsdp_size=2, tp_size=4, pp_size=2, ep_size=2, cp_size=1, microbatches=8)
    profile = calculate_memory_and_scaling(moe70b, full_cfg, batch_size_per_gpu=1)
    
    print(f"{'Axis':<6} | {'Network Domain':<24} | {'Collective Type':<32} | {'MB / Step'}")
    print("-" * 75)
    for axis, domain, desc, mb in profile["communication"]["outer_to_inner_hierarchy"]:
        print(f"{axis:<6} | {domain:<24} | {desc:<32} | {mb:>8.2f} MB")
    print("-" * 75)
    print("Conclusion: As you move inward from PP to TP, frequency increases, latency")
    print("sensitivity skyrockets, and blocking collectives require ultra-high bandwidth (NVLink).")


if __name__ == "__main__":
    main()
