"""Maintainer check: every starter is importable and fails ONLY on its TODOs.

    uv run python scripts/check_starters.py

For each lab, a temporary copy of the repository is made with the starter
copied over the reference module, then the lab's tests are run. A healthy
starter makes pytest report test failures (exit code 1) caused by
NotImplementedError — never import or collection errors (exit code 2+).
The reference implementation passing the same tests is checked by the
ordinary `pytest tests/` run.

Multi-process tests are skipped here (`-m "not distributed"`): the
single-process tests already reach every TODO, and the labs run in parallel.
Extension tests are included (`--extensions`) so that their TODOs are checked too.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# A paste-in scaffold is not a drop-in module: students paste its methods into
# their own model. To check it, run its methods in place of the reference ones.
PASTE_IN = "paste-in"
_PASTE_PATCH = """
_ns = {}
exec(compile(open(__file__.replace("model.py", "_scaffold.py")).read(), "_scaffold", "exec"), _ns)
CausalSelfAttention.forward = _ns["CausalSelfAttention"].forward
CausalSelfAttention.forward_with_cache = _ns["CausalSelfAttention"].forward_with_cache
TransformerBlock.forward_with_cache = _ns["TransformerBlock"].forward_with_cache
MiniGPT.forward = _ns["MiniGPT"].forward
"""

LABS = {
    "Module 1 — Transformer": ({"starter/01_transformer.py": "nanolm/model.py"}, "test_module_01_model.py"),
    "Module 2 — training loop": (
        {"starter/02_training/optim.py": "nanolm/optim.py", "starter/02_training/train.py": "nanolm/train.py"},
        "test_module_02_optim.py test_module_02_baseline.py",
    ),
    "Module 3 — data pipeline": ({"starter/03_data.py": "nanolm/data.py"}, "test_module_03_data.py"),
    "Module 4 — profiling": ({"starter/04_profile_utils.py": "nanolm/profile_utils.py"}, "test_module_04_profile.py"),
    "Module 5 — manual DDP": ({"starter/05_ddp_manual.py": "nanolm/ddp_manual.py"}, "test_module_05_ddp.py"),
    "Module 6 — sharded AdamW": ({"starter/06_zero.py": "nanolm/zero.py"}, "test_module_06_zero.py"),
    "Module 7 — tensor parallelism": (
        {"starter/07_tensor_parallel.py": "nanolm/tensor_parallel.py"},
        "test_module_07_tp.py",
    ),
    "Module 9 — KV-cached generation": ({"starter/09_generate.py": "nanolm/generate.py"}, "test_module_09_kv_cache.py"),
    "Module 9 — KV-cache scaffold": ({PASTE_IN: "starter/09_kv_cache_methods.py"}, "test_module_09_kv_cache.py"),
}


def check(name: str, overlay: dict, test_files: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for item in ("nanolm", "tests", "scripts", "starter"):
            shutil.copytree(ROOT / item, work / item, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
        for source, target in overlay.items():
            if source == PASTE_IN:
                shutil.copy(ROOT / target, work / "nanolm/_scaffold.py")
                with open(work / "nanolm/model.py", "a") as model_file:
                    model_file.write(_PASTE_PATCH)
            else:
                shutil.copy(ROOT / source, work / target)
        # Subprocesses (torchrun workers) must import the overlaid copy, not the
        # editable install of the reference package.
        env = dict(os.environ, PYTHONPATH=str(work))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *(f"tests/{name}" for name in test_files.split()),
             "-q", "-p", "no:cacheprovider", "-m", "not distributed", "--extensions"],
            cwd=work, env=env, capture_output=True, text=True, timeout=600,
        )
    output = result.stdout + result.stderr
    healthy = result.returncode == 1 and "NotImplementedError" in output
    summary = output.strip().splitlines()[-1] if output.strip() else "(no output)"
    report = f"{'ok  ' if healthy else 'FAIL'} {name:34s} exit={result.returncode}  {summary}"
    if not healthy:
        report += "\n" + output[-3000:]
    return healthy, report


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda lab: check(lab[0], *lab[1]), LABS.items()))
    for _, report in outcomes:
        print(report)
    results = [healthy for healthy, _ in outcomes]
    if not all(results):
        raise SystemExit("Some starters are broken: see output above.")
    print("All starters import cleanly and fail only on their TODOs.")


if __name__ == "__main__":
    main()
