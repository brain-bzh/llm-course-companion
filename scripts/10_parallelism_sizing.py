"""Session 10 — Context and pipeline parallelism.

Demonstrates:
- Analytical calculator for 3D/4D parallelism (DP x TP x PP x CP).
- Pipeline bubble overhead calculation under 1F1B schedule.
- Per-GPU VRAM estimation across cluster topologies.
"""

from minilm.parallelism_calc import ModelSpecs, ParallelismConfig, calculate_memory_and_scaling

def main():
    print("=== Session 10: Multidimensional Parallelism & Cluster Sizing ===")

    # Case: 7B LLaMA-style model
    llama7b = ModelSpecs(
        name="LLaMA-7B",
        n_layer=32,
        n_head=32,
        d_model=4096,
        seq_len=4096,
        vocab_size=32000,
    )

    scenarios = [
        ("Single Node (8 GPUs) - pure DDP / FSDP", ParallelismConfig(dp_size=8, tp_size=1, pp_size=1, cp_size=1)),
        ("Single Node (8 GPUs) - TP=8", ParallelismConfig(dp_size=1, tp_size=8, pp_size=1, cp_size=1)),
        ("Cluster (32 GPUs) - DP=4, TP=4, PP=2", ParallelismConfig(dp_size=4, tp_size=4, pp_size=2, cp_size=1, microbatches=8)),
        ("Long Context (64 GPUs) - DP=4, TP=4, PP=2, CP=2", ParallelismConfig(dp_size=4, tp_size=4, pp_size=2, cp_size=2, microbatches=16)),
    ]

    for label, cfg in scenarios:
        res = calculate_memory_and_scaling(llama7b, cfg, batch_size_per_gpu=1)
        print(f"\nScenario: {label}")
        print(f"  Total GPUs:                  {res['total_gpus']}")
        print(f"  Pipeline Bubble:             {res['pipeline_bubble_percent']}%")
        print(f"  Weights per GPU:             {res['weights_per_gpu_gb']} GB")
        print(f"  Gradients per GPU:           {res['grads_per_gpu_gb']} GB")
        print(f"  Optimizer per GPU (ZeRO-1):  {res['optimizer_per_gpu_gb']} GB")
        print(f"  Activations per GPU:         {res['activations_per_gpu_gb']} GB")
        print(f"  -> Estimated Peak VRAM:      {res['estimated_peak_memory_per_gpu_gb']} GB")

if __name__ == "__main__":
    main()
