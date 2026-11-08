"""Shared helpers: run a companion script on several CPU ranks with Gloo.

Students launch multi-process exercises with torchrun (Linux and macOS):

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/<script>.py

The tests instead start the worker processes directly, with the same
environment variables torchrun would set (RANK, WORLD_SIZE, MASTER_ADDR, ...).
`init_process_group` then uses its standard env:// rendezvous. This
- skips torchrun's agent process, saving one Python + torch start-up per test;
- works on native Windows, where torchrun 2.9's launcher store requires libuv
  and ignores USE_LIBUV=0 (the env:// rendezvous respects it).

`test_torchrun_launcher.py` still checks the exact student command once.
Tests that start several processes are marked `distributed`.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker_env(rank: int, nproc: int, port: int) -> dict:
    return dict(
        os.environ,
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(nproc), LOCAL_WORLD_SIZE=str(nproc),
        GROUP_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        OMP_NUM_THREADS="1", PYTHONUTF8="1", USE_LIBUV="0",
    )


def run_ranks(script: str, *args: str, nproc: int = 2, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run scripts/<script> as `nproc` ranks; return rank 0's stdout and every rank's stderr."""
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            cwd=ROOT, env=_worker_env(rank, nproc, port),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for rank in range(nproc)
    ]
    deadline = time.monotonic() + timeout
    outputs = []
    try:
        for proc in procs:
            outputs.append(proc.communicate(timeout=max(1.0, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail(f"{script} did not finish within {timeout}s (a rank is probably waiting on a collective)")
    returncode = next((p.returncode for p in procs if p.returncode != 0), 0)
    stderr = "\n".join(f"--- rank {rank} stderr ---\n{err}" for rank, (_, err) in enumerate(outputs) if err.strip())
    return subprocess.CompletedProcess(procs[0].args, returncode, outputs[0][0], stderr)


@pytest.fixture
def torchrun():
    """Return run_ranks: launches `scripts/<script>` as several Gloo ranks."""
    return run_ranks


def pytest_collection_modifyitems(items):
    for item in items:
        if "torchrun" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.distributed)
