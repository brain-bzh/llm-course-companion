"""Serving system simulation: continuous batching and paged KV memory.

Covers:
- Module 10: Serving systems (request lifecycle, static vs continuous batching,
  reserved vs paged KV allocation, preemption, TTFT, ITL, throughput).

A deliberately small discrete-time model. Each engine iteration costs

    step_latency_sec + prefill_sec_per_token * (prompt tokens prefilled this iteration)

so long prompts delay the first token of their own request AND stretch the
inter-token latency of every request decoding in the same iteration.
Requests are only visible to the scheduler once `arrival_time` has passed.

This is a teaching model, not a performance predictor: it has no attention-cost
growth with context, no chunked prefill, no prefix caching, and no GPU.
"""

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional


@dataclass
class Request:
    req_id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    generated_tokens: int = 0
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    preemptions: int = 0

    @property
    def ttft(self) -> float:
        """Time to first token, measured from arrival (includes queueing)."""
        if self.first_token_time is None:
            raise ValueError(f"request {self.req_id} produced no token")
        return self.first_token_time - self.arrival_time

    @property
    def mean_itl(self) -> Optional[float]:
        """Mean inter-token latency after the first token (None for one-token outputs)."""
        if len(self.token_times) < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)


class PagedMemoryAllocator:
    """KV blocks of `block_size` tokens, granted on demand as a sequence grows."""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.req_to_blocks: Dict[int, List[int]] = {}

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def allocate_tokens(self, req_id: int, total_tokens: int) -> bool:
        """Grow `req_id` to hold `total_tokens`; return False (and change nothing) if full."""
        current = self.req_to_blocks.setdefault(req_id, [])
        missing = self.blocks_for(total_tokens) - len(current)
        if missing <= 0:
            return True
        if len(self.free_blocks) < missing:
            if not current:
                del self.req_to_blocks[req_id]
            return False
        current.extend(self.free_blocks.pop(0) for _ in range(missing))
        return True

    def free_request(self, req_id: int) -> None:
        self.free_blocks.extend(self.req_to_blocks.pop(req_id, []))

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self.free_blocks)


class ContinuousBatchingSimulator:
    """Iteration-level scheduler with configurable batching and KV allocation.

    scheduler="continuous": admit arrived requests whenever a slot and memory allow.
    scheduler="static":     admit a new batch only when the previous one has finished.
    allocation="paged":     grow each request's KV blocks one token at a time;
                            if growth fails, preempt the newest request (recompute later).
    allocation="reserve":   reserve prompt_len + max_output_len tokens at admission.
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        total_blocks: int = 64,
        block_size: int = 16,
        step_latency_sec: float = 0.01,
        prefill_sec_per_token: float = 0.0,
        scheduler: str = "continuous",
        allocation: str = "paged",
        max_output_len: int = 64,
    ):
        if scheduler not in ("continuous", "static") or allocation not in ("paged", "reserve"):
            raise ValueError("scheduler must be continuous|static and allocation paged|reserve")
        self.max_batch_size = max_batch_size
        self.step_latency = step_latency_sec
        self.prefill_cost = prefill_sec_per_token
        self.scheduler = scheduler
        self.allocation = allocation
        self.max_output_len = max_output_len
        self.allocator = PagedMemoryAllocator(total_blocks, block_size)
        self.waiting_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.completed_requests: List[Request] = []
        self.current_time = 0.0
        self.peak_running = 0
        self.peak_blocks = 0
        self.preemptions = 0

    def add_request(self, req: Request) -> None:
        if self.allocation == "reserve" and req.output_len > self.max_output_len:
            raise ValueError("reserve allocation needs output_len <= max_output_len")
        self.waiting_queue.append(req)
        self.waiting_queue.sort(key=lambda r: r.arrival_time)

    def _admission_tokens(self, req: Request) -> int:
        if self.allocation == "reserve":
            return req.prompt_len + self.max_output_len
        return req.prompt_len + req.generated_tokens

    def _admit(self) -> List[Request]:
        if self.scheduler == "static" and self.running_batch:
            return []
        admitted = []
        for req in list(self.waiting_queue):
            if req.arrival_time > self.current_time or len(self.running_batch) >= self.max_batch_size:
                break  # first-come first-served: do not skip ahead of the queue head
            if not self.allocator.allocate_tokens(req.req_id, self._admission_tokens(req)):
                break
            self.waiting_queue.remove(req)
            if req.start_time is None:
                req.start_time = self.current_time
            self.running_batch.append(req)
            admitted.append(req)
        return admitted

    def _preempt_newest(self) -> None:
        victim = self.running_batch.pop()  # most recently admitted
        self.allocator.free_request(victim.req_id)
        victim.preemptions += 1
        self.preemptions += 1
        victim.generated_tokens = 0  # recompute: its KV cache is gone
        victim.token_times.clear()
        victim.first_token_time = None
        self.waiting_queue.insert(0, victim)

    def step(self) -> None:
        """Execute one engine iteration (or jump forward to the next arrival)."""
        admitted = self._admit()
        if not self.running_batch:
            future = [r.arrival_time for r in self.waiting_queue if r.arrival_time > self.current_time]
            if not future:
                raise RuntimeError("a waiting request can never fit in KV memory")
            self.current_time = min(future)
            return

        prefill_tokens = sum(r.prompt_len for r in admitted)
        self.current_time += self.step_latency + self.prefill_cost * prefill_tokens
        self.peak_running = max(self.peak_running, len(self.running_batch))

        for req in list(self.running_batch):
            if req not in self.running_batch:
                continue  # preempted earlier in this iteration
            if self.allocation == "paged":
                needed = req.prompt_len + req.generated_tokens + 1
                while not self.allocator.allocate_tokens(req.req_id, needed):
                    if len(self.running_batch) == 1:
                        raise RuntimeError(f"request {req.req_id} needs more KV blocks than exist")
                    self._preempt_newest()
                    if req not in self.running_batch:
                        break
                if req not in self.running_batch:
                    continue
            req.generated_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = self.current_time
            req.token_times.append(self.current_time)
            if req.generated_tokens >= req.output_len:
                req.finish_time = self.current_time
                self.allocator.free_request(req.req_id)
                self.completed_requests.append(req)
                self.running_batch.remove(req)
        self.peak_blocks = max(self.peak_blocks, self.allocator.used_blocks)

    def run_all(self) -> Dict[str, Any]:
        """Run until every request has finished and summarize the trace."""
        while self.waiting_queue or self.running_batch:
            self.step()
        elapsed = self.current_time  # the simulation clock starts at 0
        done = sorted(self.completed_requests, key=lambda r: r.req_id)
        ttfts = sorted(r.ttft for r in done)
        itls = [r.mean_itl for r in done if r.mean_itl is not None]
        output_tokens = sum(r.generated_tokens for r in done)
        return {
            "scheduler": self.scheduler,
            "allocation": self.allocation,
            "total_requests": len(done),
            "total_sim_time_sec": round(elapsed, 4),
            "mean_ttft_ms": round(1000 * sum(ttfts) / len(ttfts), 2),
            "p50_ttft_ms": round(1000 * median(ttfts), 2),
            "max_ttft_ms": round(1000 * ttfts[-1], 2),
            "mean_itl_ms": round(1000 * sum(itls) / len(itls), 2) if itls else None,
            "output_tokens_per_sec": round(output_tokens / elapsed, 1),
            "peak_running_requests": self.peak_running,
            "peak_kv_blocks": self.peak_blocks,
            "preemptions": self.preemptions,
        }
