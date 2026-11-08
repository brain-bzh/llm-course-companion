"""The exact multi-process command students use, run once (Linux and macOS)."""

import os
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.mark.distributed
@pytest.mark.skipif(sys.platform == "win32", reason="torchrun 2.9 needs libuv on Windows; use WSL2 or Linux")
def test_student_torchrun_command_runs_two_ranks():
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
           "--nproc_per_node=2", str(ROOT / "scripts" / "07_tensor_parallel.py")]
    result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
