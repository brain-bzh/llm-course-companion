"""Module 7: actual Gloo TP forward/backward equivalence on CPU.

uv run torchrun --standalone --nproc_per_node=2 scripts/07_tensor_parallel.py
"""
import os
import sys
import torch
import torch.distributed as dist
from nanolm.tensor_parallel import verify_sharded_mlp


def main():
    torch.set_num_threads(1)
    if sys.platform == 'darwin':
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('gloo')
    try:
        verify_sharded_mlp()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('PASS: TP outputs, input gradients and both parameter shards match the reference.')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
