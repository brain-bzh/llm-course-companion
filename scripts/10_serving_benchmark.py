"""Module 10: one workload, four engine policies, in a deterministic simulator.

    uv run python scripts/10_serving_benchmark.py
    uv run python scripts/10_serving_benchmark.py --blocks 16 --trace

Compares static vs continuous batching and reserved vs paged KV allocation
on the same staggered arrivals. Times are simulated, not measured on hardware.
"""

import argparse

from nanolm.serving_sim import ContinuousBatchingSimulator, Request


WORKLOAD = [  # (arrival s, prompt tokens, output tokens)
    (0.00, 32, 16), (0.01, 48, 32), (0.02, 16, 8), (0.03, 64, 24),
    (0.05, 20, 12), (0.08, 40, 20), (0.10, 16, 16), (0.12, 32, 8),
]


def run(scheduler, allocation, args, trace=False):
    sim = ContinuousBatchingSimulator(
        max_batch_size=args.batch, total_blocks=args.blocks, block_size=16,
        step_latency_sec=0.015, prefill_sec_per_token=0.0002,
        scheduler=scheduler, allocation=allocation, max_output_len=32,
    )
    for i, (arrival, prompt, output) in enumerate(WORKLOAD, start=1):
        sim.add_request(Request(i, arrival, prompt, output))
    if trace:
        print(f"\ntrace: {scheduler} batching, {allocation} KV")
        print(f"{'t (ms)':>8s}  running (id:generated/output)       waiting  KV blocks")
        while sim.waiting_queue or sim.running_batch:
            sim.step()
            running = " ".join(f"{r.req_id}:{r.generated_tokens}/{r.output_len}" for r in sim.running_batch)
            waiting = [r.req_id for r in sim.waiting_queue if r.arrival_time <= sim.current_time]
            print(f"{1000 * sim.current_time:8.1f}  {running:34s} {str(waiting):8s} {sim.allocator.used_blocks:3d}")
    return sim.run_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=4, help="maximum running requests")
    parser.add_argument("--blocks", type=int, default=24, help="KV blocks of 16 tokens")
    parser.add_argument("--trace", action="store_true", help="print the continuous+paged iteration trace")
    args = parser.parse_args()
    print(f"workload: {len(WORKLOAD)} requests; batch <= {args.batch}; {args.blocks} KV blocks x 16 tokens")
    cols = {"mean_ttft_ms": "TTFT mean", "max_ttft_ms": "TTFT max", "mean_itl_ms": "ITL mean",
            "output_tokens_per_sec": "out tok/s", "peak_running_requests": "peak run",
            "peak_kv_blocks": "peak blocks", "preemptions": "preempt"}
    print("times in ms of simulated clock")
    print(f"\n{'policy':22s}" + "".join(f"{label:>12s}" for label in cols.values()))
    for scheduler in ("static", "continuous"):
        for allocation in ("reserve", "paged"):
            m = run(scheduler, allocation, args)
            print(f"{scheduler + ' + ' + allocation:22s}" + "".join(f"{m[c]:12}" for c in cols))
    if args.trace:
        run("continuous", "paged", args, trace=True)


if __name__ == "__main__":
    main()
