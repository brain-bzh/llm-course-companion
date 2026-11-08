"""Shared helpers: launch a companion script on several CPU ranks with Gloo.

The launch command is the one students use, on Linux, macOS and Windows:

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/<script>.py

Platform notes:
- `--local-addr=127.0.0.1` keeps the launcher's rendezvous on loopback instead of
  a reverse-resolved hostname (which can hang on macOS laptops and some Wi-Fi/VPNs).
- Workers call `nanolm.distributed.configure_local_gloo()` to pin Gloo to loopback.
- Windows PyTorch wheels lack libuv for the launcher's TCPStore: set USE_LIBUV=0.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_torchrun(script: str, *args: str, nproc: int = 2, timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUTF8="1")
    if sys.platform == "win32":
        env.setdefault("USE_LIBUV", "0")
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
        f"--nproc_per_node={nproc}", str(ROOT / "scripts" / script), *args,
    ]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def torchrun():
    """Return a function that runs `scripts/<script>` under torchrun and returns the process."""
    return _run_torchrun
