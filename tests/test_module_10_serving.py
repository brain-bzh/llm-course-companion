"""Tests for Module 10 — the serving simulator's scheduling invariants."""

import pytest

from nanolm.serving_sim import ContinuousBatchingSimulator, PagedMemoryAllocator, Request


WORKLOAD = [(0.00, 32, 16), (0.01, 48, 32), (0.02, 16, 8), (0.03, 64, 24),
            (0.05, 20, 12), (0.08, 40, 20), (0.10, 16, 16), (0.12, 32, 8)]


def simulate(**kwargs):
    options = dict(max_batch_size=4, total_blocks=24, block_size=16, step_latency_sec=0.015,
                   prefill_sec_per_token=0.0002, max_output_len=32)
    options.update(kwargs)
    sim = ContinuousBatchingSimulator(**options)
    for i, (arrival, prompt, output) in enumerate(WORKLOAD, start=1):
        sim.add_request(Request(i, arrival, prompt, output))
    return sim, sim.run_all()


@pytest.mark.parametrize("scheduler", ["static", "continuous"])
@pytest.mark.parametrize("allocation", ["reserve", "paged"])
def test_every_request_finishes_after_it_arrives(scheduler, allocation):
    sim, metrics = simulate(scheduler=scheduler, allocation=allocation, total_blocks=10, max_batch_size=8)
    assert metrics["total_requests"] == len(WORKLOAD)
    for r in sim.completed_requests:
        assert r.start_time >= r.arrival_time
        assert r.generated_tokens == r.output_len == len(r.token_times)
        assert r.ttft > 0
    assert sim.allocator.used_blocks == 0  # every block returned


def test_continuous_batching_reduces_queueing_delay():
    _, static = simulate(scheduler="static")
    _, continuous = simulate(scheduler="continuous")
    assert continuous["mean_ttft_ms"] < static["mean_ttft_ms"]
    assert continuous["output_tokens_per_sec"] > static["output_tokens_per_sec"]


def test_paged_allocation_admits_more_requests_under_tight_memory():
    _, reserve = simulate(allocation="reserve", total_blocks=10, max_batch_size=8)
    _, paged = simulate(allocation="paged", total_blocks=10, max_batch_size=8)
    assert paged["peak_running_requests"] > reserve["peak_running_requests"]
    assert paged["preemptions"] >= 1  # optimism has a price


def test_prefill_stretches_the_iteration_that_admits_a_prompt():
    sim = ContinuousBatchingSimulator(step_latency_sec=0.01, prefill_sec_per_token=0.001)
    sim.add_request(Request(1, 0.0, prompt_len=100, output_len=2))
    sim.step()
    assert sim.current_time == pytest.approx(0.01 + 0.1)
    sim.step()
    assert sim.current_time == pytest.approx(0.12)


def test_allocator_is_all_or_nothing():
    allocator = PagedMemoryAllocator(total_blocks=2, block_size=4)
    assert allocator.allocate_tokens(1, 8)
    assert not allocator.allocate_tokens(2, 1)
    assert 2 not in allocator.req_to_blocks
    allocator.free_request(1)
    assert allocator.used_blocks == 0
