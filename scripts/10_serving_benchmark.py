"""Module 11 — Serving systems.

Demonstrates:
- Continuous (iteration-level) batching vs static batching.
- Paged KV memory block allocation.
- Measuring Time To First Token (TTFT), Inter-Token Latency (ITL), and Request Throughput.
"""

from minilm.serving_sim import Request, ContinuousBatchingSimulator

def main():
    print("=== Module 11: Continuous Batching & Serving Benchmark ===")

    sim = ContinuousBatchingSimulator(
        max_batch_size=4,
        total_blocks=64,
        block_size=16,
        step_latency_sec=0.015,  # 15 ms per token step
    )

    # Workload of 8 arriving requests with variable lengths
    workload = [
        Request(req_id=1, arrival_time=0.00, prompt_len=32, output_len=16),
        Request(req_id=2, arrival_time=0.01, prompt_len=48, output_len=32),
        Request(req_id=3, arrival_time=0.02, prompt_len=16, output_len=8),
        Request(req_id=4, arrival_time=0.03, prompt_len=64, output_len=24),
        Request(req_id=5, arrival_time=0.05, prompt_len=20, output_len=12),
        Request(req_id=6, arrival_time=0.08, prompt_len=40, output_len=20),
        Request(req_id=7, arrival_time=0.10, prompt_len=16, output_len=16),
        Request(req_id=8, arrival_time=0.12, prompt_len=32, output_len=8),
    ]

    for req in workload:
        sim.add_request(req)

    print(f"Simulating continuous batching scheduler for {len(workload)} requests...")
    metrics = sim.run_all()

    print("\n--- Serving Benchmark Report ---")
    print(f"Total Requests Completed:   {metrics['total_requests']}")
    print(f"Total Time:                 {metrics['total_sim_time_sec']} s")
    print(f"Mean TTFT:                  {metrics['mean_ttft_ms']} ms")
    print(f"Mean Inter-Token Latency:   {metrics['mean_itl_ms']} ms")
    print(f"Request Throughput:         {metrics['request_throughput_req_per_sec']} req/s")
    print(f"Token Throughput:           {metrics['token_throughput_tokens_per_sec']} tokens/s")

if __name__ == "__main__":
    main()
