"""Module 8: persistent state and device counting, not a peak-VRAM predictor."""
from nanolm.parallelism_calc import ModelSpecs, ParallelismConfig, calculate_memory_and_scaling


def main():
    specs = ModelSpecs("Teaching MoE (not a production architecture)", 32, 32, 4096, 4096,
                       vocab_size=32000, n_experts=8, top_k=2)
    scenarios = [
        ("TP within one 8-device group", ParallelismConfig(tp_size=8)),
        ("32 devices: D=4, TP=4, PP=2", ParallelismConfig(dp_size=4, tp_size=4, pp_size=2)),
        ("Same 32 devices; EP=4 subdivides D", ParallelismConfig(dp_size=4, tp_size=4, pp_size=2, ep_size=4)),
        ("Same 32 devices; F=4 shards all states instead", ParallelismConfig(dp_size=4, fsdp_size=4, tp_size=4, pp_size=2)),
    ]
    for label, cfg in scenarios:
        result = calculate_memory_and_scaling(specs, cfg, batch_size_per_gpu=1)
        print(f"\n{label}")
        print(f"Devices: {result['total_gpus']}; global tokens/update: {result['global_tokens_per_step']:,}")
        print(f"Ideal pipeline idle fraction: {100 * result['pipeline_bubble_fraction']:.2f}%")
        print(f"Average persistent state/rank: {result['persistent_state_bytes'] / 2**30:.2f} GiB")
        print(f"Excluded: {result['exclusions']}")
    print("\nFit requires an activation/buffer budget and a measured peak on the actual implementation.")


if __name__ == "__main__":
    main()
