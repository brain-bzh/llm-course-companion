"""ZeRO-style sharded AdamW written by hand with two collectives.

Covers:
- Module 6: FSDP and ZeRO (optimizer-state sharding, reduce-scatter,
  all-gather, padding, and global gradient clipping under sharding).

Every rank keeps a full parameter replica for forward and backward, exactly as
in DDP. The difference is the update:

    full gradients ──reduce-scatter──▶ averaged gradient for MY slice
    AdamW on MY slice only            (moments exist only for my slice)
    updated slices ──all-gather────▶ full parameters on every rank

An all-reduce is mathematically a reduce-scatter followed by an all-gather, so
this saves optimizer-state memory without adding communication volume. This is
ZeRO stage 1 (sharded optimizer state); freeing the full gradients after the
reduce-scatter gives the persistent-memory behavior of stage 2. Stage 3 / FSDP
additionally shards the parameters themselves: see scripts/06_fsdp2.py.

Works without an initialized process group (one rank owns everything), so it
can be checked against torch.optim.AdamW in a single process.
"""

import math

import torch
import torch.distributed as dist


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class ShardedAdamW:
    """AdamW whose moments and update are partitioned across data-parallel ranks.

    The parameters are viewed as one flat vector of `numel` elements, padded with
    zeros to `shard_size * world_size`. Rank r owns elements
    [r * shard_size, (r + 1) * shard_size). Weight decay is decoupled and applied
    to every parameter, matching torch.optim.AdamW with a single parameter group.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, max_grad_norm=None):
        unique, seen = [], set()
        for p in params:
            if p.requires_grad and id(p) not in seen:  # tied weights appear once
                unique.append(p)
                seen.add(id(p))
        self.params = unique
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.max_grad_norm = max_grad_norm
        self.rank, self.world_size = _world()
        self.numel = sum(p.numel() for p in self.params)
        self.shard_size = math.ceil(self.numel / self.world_size)
        self.start = self.rank * self.shard_size
        reference = self.params[0]
        self.exp_avg = torch.zeros(self.shard_size, dtype=torch.float32, device=reference.device)
        self.exp_avg_sq = torch.zeros_like(self.exp_avg)
        self.step_count = 0
        self.last_grad_norm = None

    # ----- provided helpers -------------------------------------------------
    def _flatten(self, tensors) -> torch.Tensor:
        """Concatenate tensors into one zero-padded vector of shard_size * world_size."""
        flat = torch.cat([t.reshape(-1).float() for t in tensors])
        padding = self.shard_size * self.world_size - flat.numel()
        return torch.nn.functional.pad(flat, (0, padding))

    def _write_back(self, flat: torch.Tensor) -> None:
        """Copy a full flat vector back into the parameter tensors."""
        offset = 0
        for p in self.params:
            p.data.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def state_bytes(self) -> int:
        """Bytes of optimizer state held by THIS rank."""
        return (self.exp_avg.numel() + self.exp_avg_sq.numel()) * self.exp_avg.element_size()

    # ----- the three steps of a sharded update --------------------------------
    def _reduce_scatter_mean(self, flat_grad: torch.Tensor) -> torch.Tensor:
        """Return the cross-rank MEAN gradient for this rank's slice only."""
        if self.world_size == 1:
            return flat_grad.clone()
        shard = torch.empty(self.shard_size, dtype=flat_grad.dtype, device=flat_grad.device)
        dist.reduce_scatter_tensor(shard, flat_grad, op=dist.ReduceOp.SUM)
        return shard.div_(self.world_size)

    def _clip_(self, grad_shard: torch.Tensor) -> None:
        """Extension: clip by the GLOBAL norm, although each rank holds one slice."""
        squared = grad_shard.square().sum()
        if self.world_size > 1:
            dist.all_reduce(squared, op=dist.ReduceOp.SUM)
        total_norm = squared.sqrt()
        self.last_grad_norm = total_norm.item()
        coefficient = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        grad_shard.mul_(coefficient)

    def _adamw_update(self, param_shard: torch.Tensor, grad_shard: torch.Tensor) -> None:
        """Apply one AdamW step in place to this rank's slice."""
        beta1, beta2 = self.betas
        t = self.step_count
        param_shard.mul_(1 - self.lr * self.weight_decay)
        self.exp_avg.mul_(beta1).add_(grad_shard, alpha=1 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t
        denominator = (self.exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        param_shard.addcdiv_(self.exp_avg, denominator, value=-self.lr / bias_correction1)

    def _all_gather(self, param_shard: torch.Tensor) -> torch.Tensor:
        """Collect every rank's updated slice into one full flat vector."""
        if self.world_size == 1:
            return param_shard
        full = torch.empty(self.shard_size * self.world_size, dtype=param_shard.dtype, device=param_shard.device)
        dist.all_gather_into_tensor(full, param_shard)
        return full

    @torch.no_grad()
    def step(self) -> None:
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        grad_shard = self._reduce_scatter_mean(self._flatten(grads))
        self.zero_grad()  # full gradients are no longer needed: ZeRO-2 memory behavior
        if self.max_grad_norm is not None:
            self._clip_(grad_shard)
        self.step_count += 1
        end = self.start + self.shard_size
        param_shard = self._flatten([p.data for p in self.params])[self.start : end].clone()
        self._adamw_update(param_shard, grad_shard)
        self._write_back(self._all_gather(param_shard))
