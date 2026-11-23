# NanoLM Companion Code

Companion codebase for [**Training and Scaling Language Models: From First Principles to Efficient Serving**](https://brain-bzh.github.io/llm-course/), designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).

This repository provides an ultra-lean, pedagogical implementation of an end-to-end language modeling system. It is designed to be completely inspectable without heavy third-party framework abstractions.

---

## 1. Design & Dependency Philosophy

- **Minimal Dependencies**: The entire codebase runs with only **three runtime dependencies**:
  - `torch` (Tensor operations, automatic differentiation, neural network primitives)
  - `numpy` (Binary memory mapping and numerical helpers)
  - `tiktoken` (Standard Byte-Pair Encoding tokenizer for GPT-style models)
  - *(`pytest` for test verification)*
- **Self-Contained & Isolated**: This repository has its own `pyproject.toml` and can be cloned independently. Course maintainers also pin it as the `companion/` submodule of [`brain-bzh/llm-course`](https://github.com/brain-bzh/llm-course).
- **Python Version**: The companion uses CPython 3.12. The version file and project metadata keep `uv sync`, `uv run`, and CI on the same supported minor version; PyTorch 2.9.1 does not support `torch.compile` on Python 3.14.
- **Intentional PyTorch Pin**: `torch==2.9.1` is the newest supported line with a prebuilt [FlashAttention v2.8.3.post1](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1) wheel for the accelerated labs. Those wheels specifically target Linux, CPython 3.12, CUDA 13, and the C++11 ABI. The base `uv sync` remains cross-platform and does not install FlashAttention; do not upgrade PyTorch without checking or rebuilding that optional accelerated path.
- **Pedagogical Transparency**: Every mechanism (attention, causal masking, weight decay grouping, token packing, KV caching, tensor parallelism, serving queues, MoE routing) is written in readable, well-commented PyTorch.

---

## 2. Quickstart

Navigate to the cloned repository and install dependencies using `uv`:

```bash
# Clone the public companion repository
git clone https://github.com/brain-bzh/llm-course-companion.git
cd llm-course-companion

# Install or select Python 3.12, then sync minimal dependencies
uv python install

# Sync the project environment
uv sync

# Run the test suite
uv run pytest tests/ -v
```

### Supported platforms

Every lab runs on a laptop CPU. Distributed labs use two or more CPU processes
with PyTorch's Gloo backend; GPUs are only needed for the optional measurement
extensions.

| Platform | Status | Notes |
| --- | --- | --- |
| Linux (x86-64, aarch64) | Supported, tested in CI | Also the setup for teaching GPUs |
| macOS (Apple silicon, Intel) | Supported, tested in CI | |
| Windows | Use **WSL2** (Ubuntu) or the school's Linux machines | Single-process labs work natively; `torchrun` does not (see below) |

Launch every multi-process exercise with the same command:

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py
```

`--local-addr=127.0.0.1` keeps the launcher on the loopback address; the
scripts pin Gloo to the loopback interface (`nanolm.distributed.configure_local_gloo`).
On native Windows, `torchrun` from PyTorch 2.9 fails with `use_libuv was
requested but PyTorch was built without libuv support`, and setting `USE_LIBUV=0`
does not fix it for the launcher. Run the multi-process labs in WSL2 or on the
school's Linux machines. If a two-rank command hangs for more than
a minute, see the troubleshooting section of the
[companion guide](https://brain-bzh.github.io/llm-course/companion/#troubleshooting-distributed-runs).

On a compatible Linux GPU environment (CPython 3.12, CUDA 13), install the
optional prebuilt FlashAttention path with:

```bash
uv sync --extra flash-attention
```

The extra uses the Torch version from the runtime environment while resolving
FlashAttention's build dependency, matching the setup used in the REVE project.

### How the labs work

`nanolm/` always contains a complete, tested **reference** implementation, so
every later lab runs from a fresh clone. For a hands-on lab, create a branch
and copy the lab's starter over the reference module:

```bash
git switch -c session-14-ddp
cp starter/05_ddp_manual.py nanolm/ddp_manual.py
uv run pytest tests/test_module_05_ddp.py -q   # fails with NotImplementedError until you finish
```

Lab 1 records the starting revision in `.course-reference-revision`. To recover
the reference for a file later, commit your work first, then run
`git show "$(cat .course-reference-revision):nanolm/<file>.py" > nanolm/<file>.py`.
The lab guides say which file each session replaces.

The [continuous project](https://brain-bzh.github.io/llm-course/project/) page
describes how these labs build one system over the course, the milestones and
their minimum evidence, and the lab log every student keeps.

## 3. Module-by-Module Topic Coverage

| Module | Hands-on starter (you implement) | Reference modules | Runnable scripts | Exit criterion |
| :---: | :--- | :--- | :--- | :--- |
| **1** [Transformer](https://brain-bzh.github.io/llm-course/companion/01-transformer/) | `starter/01_transformer.py` | `nanolm/model.py`, `nanolm/gpt2.py` | `01_gpt2_parity.py` | Your model reproduces official GPT-2 logits |
| **2** [Training loop](https://brain-bzh.github.io/llm-course/companion/02-training-loop/) | `starter/02_training/` | `nanolm/optim.py`, `nanolm/train.py`, `nanolm/baseline.py` | `02_overfit.py`, `02_train_baseline.py` | Session 7: one-batch loss `< 0.1`; Session 8: exact checkpoint resume |
| **3** [Data pipeline](https://brain-bzh.github.io/llm-course/companion/03-data-pipeline/) | `starter/03_data.py` | `nanolm/data.py`, `nanolm/tokenizer.py` | `03_prepare_dataset.py` | Disjoint splits, visible boundaries, shifted batches |
| **4** [Single-GPU performance](https://brain-bzh.github.io/llm-course/companion/04-single-gpu/) | `starter/04_profile_utils.py` (+ SDPA branch in your model) | `nanolm/profile_utils.py` | `04_check_execution.py`, `04_single_gpu_perf.py` | Predicted FLOPs/memory; one controlled ablation |
| **5** [DDP](https://brain-bzh.github.io/llm-course/companion/05-ddp/) | `starter/05_ddp_manual.py` | `nanolm/ddp_manual.py`, `nanolm/distributed.py` | `05_manual_ddp.py`, `05_train_ddp.py` | Hand-written DDP equals one global-batch run |
| **6** [FSDP and ZeRO](https://brain-bzh.github.io/llm-course/companion/06-fsdp/) | `starter/06_zero.py` | `nanolm/zero.py`, `nanolm/fsdp_utils.py` | `06_zero_manual.py`, `06_fsdp2.py`, `06_checkpoint_formats.py` | Sharded AdamW and FSDP2 match the reference; measured per-rank state |
| **7** [Tensor parallelism](https://brain-bzh.github.io/llm-course/companion/07-tensor-parallelism/) | `starter/07_tensor_parallel.py` | `nanolm/tensor_parallel.py` | `07_tensor_parallel.py` | Two-rank forward and backward equal the dense MLP |
| **8** [CP, PP and EP](https://brain-bzh.github.io/llm-course/companion/08-context-pipeline/) | edit `MODEL`/`LAYOUT` in the script | `nanolm/parallelism_calc.py` | `08_parallelism_layout.py`, `08_parallelism_sizing.py` | Defended rank layout with stated assumptions |
| **9** [KV cache](https://brain-bzh.github.io/llm-course/companion/09-kv-cache/) | `starter/09_generate.py`, `starter/09_kv_cache_methods.py` (paste into your model) | `nanolm/generate.py`, `nanolm/kv_cache.py` | `09_kv_cache_bench.py` | Cached and uncached logits/tokens agree |
| **10** [Serving](https://brain-bzh.github.io/llm-course/companion/10-serving/) | optional | `nanolm/serving_sim.py` | `10_serving_benchmark.py` | Interpret TTFT/ITL/throughput across four policies |
| **11** [Frontier (optional)](https://brain-bzh.github.io/llm-course/companion/11-frontier-architectures/) | — | `nanolm/moe.py` | `11_frontier_exploration.py` | Optional modeling |

## 4. Repository Structure

```text
llm-course-companion/
├── starter/                     # TODO scaffolds: copy one over its nanolm/ module on a lab branch
│   ├── 01_transformer.py        # -> nanolm/model.py
│   ├── 02_training/             # -> nanolm/optim.py, nanolm/train.py
│   ├── 03_data.py               # -> nanolm/data.py
│   ├── 04_profile_utils.py      # -> nanolm/profile_utils.py
│   ├── 05_ddp_manual.py         # -> nanolm/ddp_manual.py
│   ├── 06_zero.py               # -> nanolm/zero.py
│   ├── 07_tensor_parallel.py    # -> nanolm/tensor_parallel.py
│   ├── 09_generate.py           # -> nanolm/generate.py
│   └── 09_kv_cache_methods.py   # paste-in: KV-cache methods for YOUR nanolm/model.py
├── nanolm/                      # Complete reference library
│   ├── model.py                 # Decoder-only GPT, explicit/SDPA attention, KV-cache path
│   ├── gpt2.py                  # Hugging Face GPT-2 weight converter
│   ├── optim.py, train.py       # AdamW grouping, schedule, update order, checkpoints
│   ├── baseline.py              # Byte-level baseline with exact resume
│   ├── tokenizer.py, data.py    # BPE, dedupe-before-split, packing, memmap loader
│   ├── data_selection.py        # Filtering, exact/near deduplication
│   ├── profile_utils.py         # FLOPs, MFU, static memory
│   ├── distributed.py           # Process-group setup, loopback Gloo configuration
│   ├── ddp_manual.py            # Hand-written data parallelism
│   ├── zero.py                  # Hand-written sharded AdamW (ZeRO-1/2)
│   ├── fsdp_utils.py            # FSDP2 wrapping and per-rank state measurement
│   ├── tensor_parallel.py       # Column/row parallel layers with autograd collectives
│   ├── parallelism_calc.py      # Multidimensional layout and state calculator
│   ├── generate.py, kv_cache.py # Cached/uncached generation, prefill/decode benchmark
│   ├── serving_sim.py           # Static/continuous batching, reserved/paged KV simulator
│   └── moe.py                   # Top-k MoE with load-balancing loss
├── scripts/                     # One runnable experiment per lab step (see table above)
│   └── check_starters.py        # Maintainer check: starters fail only on their TODOs
└── tests/                       # One test file per module; conftest.py launches torchrun
```

---

## 5. Verification & Testing

```bash
uv run pytest tests/ -v --extensions    # reference implementation: everything passes
uv run python scripts/check_starters.py # each starter imports and fails only on its TODOs
```

Tests of optional lab extensions (marked `extension`) are skipped unless pytest
runs with `--extensions`, so a student's "the module's tests pass" means the core
exercise is done. Add the flag once you attempt an extension.

Distributed tests (marked `distributed`) start real two-rank CPU jobs: the
helper in `tests/conftest.py` launches the worker processes directly, and
`tests/test_torchrun_launcher.py` checks the student `torchrun` command once.
Skip them with `-m "not distributed"` for a quick run. CI installs CPU-only
PyTorch and runs Linux on every push; macOS and Windows (best effort) run weekly
and on demand.
