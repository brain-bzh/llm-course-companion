"""Serving system simulation: Continuous batching and paged memory.

Covers:
- Session 12: Serving systems (continuous batching, TTFT, ITL, paged cache blocks, throughput).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


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

    @property
    def ttft(self) -> float:
        """Time to First Token (TTFT)."""
        if self.first_token_time is not None and self.start_time is not None:
            return self.first_token_time - self.arrival_time
        return 0.0

    @property
    def mean_itl(self) -> float:
        """Inter-Token Latency (ITL)."""
        if len(self.token_times) < 2:
            return 0.0
        intervals = [
            self.token_times[i] - self.token_times[i - 1]
            for i in range(1, len(self.token_times))
        ]
        return sum(intervals) / len(intervals)


class PagedMemoryAllocator:
    """Toy block-level memory allocator mimicking vLLM paged attention."""

    def __init__(self, total_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.req_to_blocks: Dict[int, List[int]] = {}

    def allocate_tokens(self, req_id: int, total_tokens: int) -> bool:
        blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        current_blocks = self.req_to_blocks.get(req_id, [])
        diff = blocks_needed - len(current_blocks)
        if diff <= 0:
            return True
        if len(self.free_blocks) < diff:
            return False  # Out of memory
        for _ in range(diff):
            current_blocks.append(self.free_blocks.pop(0))
        self.req_to_blocks[req_id] = current_blocks
        return True

    def free_request(self, req_id: int) -> None:
        if req_id in self.req_to_blocks:
            self.free_blocks.extend(self.req_to_blocks.pop(req_id))


class ContinuousBatchingSimulator:
    """Discrete-step simulator for continuous batching serving scheduler."""

    def __init__(
        self,
        max_batch_size: int = 4,
        total_blocks: int = 64,
        block_size: int = 16,
        step_latency_sec: float = 0.01,
    ):
        self.max_batch_size = max_batch_size
        self.step_latency = step_latency_sec
        self.allocator = PagedMemoryAllocator(total_blocks, block_size)
        self.waiting_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.completed_requests: List[Request] = []
        self.current_time = 0.0

    def add_request(self, req: Request):
        self.waiting_queue.append(req)

    def step(self):
        """Execute one iteration of continuous batching."""
        # 1. Admit waiting requests if capacity and memory allow
        admitted = []
        for req in self.waiting_queue:
            if len(self.running_batch) < self.max_batch_size:
                if self.allocator.allocate_tokens(req.req_id, req.prompt_len):
                    req.start_time = self.current_time
                    self.running_batch.append(req)
                    admitted.append(req)
            else:
                break
        for req in admitted:
            self.waiting_queue.remove(req)

        # 2. Advance time for this step
        self.current_time += self.step_latency

        # 3. Generate 1 token for each running request
        still_running = []
        for req in self.running_batch:
            req.generated_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = self.current_time
            req.token_times.append(self.current_time)

            # Check memory allocation for new token
            total_tokens = req.prompt_len + req.generated_tokens
            self.allocator.allocate_tokens(req.req_id, total_tokens)

            # Check completion
            if req.generated_tokens >= req.output_len:
                req.finish_time = self.current_time
                self.allocator.free_request(req.req_id)
                self.completed_requests.append(req)
            else:
                still_running.append(req)

        self.running_batch = still_running

    def run_all(self) -> Dict[str, Any]:
        """Run until all requests are completed."""
        start_sim_time = self.current_time
        while self.waiting_queue or self.running_batch:
            self.step()
        total_time = self.current_time - start_sim_time

        ttfts = [r.ttft for r in self.completed_requests]
        itls = [r.mean_itl for r in self.completed_requests if r.mean_itl > 0]
        total_tokens = sum(r.prompt_len + r.generated_tokens for r in self.completed_requests)

        return {
            "total_requests": len(self.completed_requests),
            "total_sim_time_sec": round(total_time, 3),
            "mean_ttft_ms": round((sum(ttfts) / len(ttfts)) * 1000, 2) if ttfts else 0.0,
            "mean_itl_ms": round((sum(itls) / len(itls)) * 1000, 2) if itls else 0.0,
            "request_throughput_req_per_sec": round(len(self.completed_requests) / total_time, 2) if total_time > 0 else 0,
            "token_throughput_tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        }
