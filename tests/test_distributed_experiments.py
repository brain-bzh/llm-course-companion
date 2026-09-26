"""Run the actual two-rank CPU exercises, including backward collectives."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('script', ['05_train_ddp.py', '07_tensor_parallel.py'])
def test_two_rank_equivalence(script):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1')
    if sys.platform == 'darwin':
        env.setdefault('GLOO_SOCKET_IFNAME', 'lo0')
        env.setdefault('MASTER_ADDR', '127.0.0.1')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--local-addr=127.0.0.1', '--nproc_per_node=2', str(root/'scripts'/script)]
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS:' in result.stdout
