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
- **Intentional PyTorch Pin**: `torch==2.9.1` is the newest supported line with a prebuilt [FlashAttention v2.8.3.post1](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1) wheel for the accelerated labs. Those wheels specifically target Linux, CPython 3.12, CUDA 13, and the C++11 ABI. The base `uv sync` remains cross-platform and does not install FlashAttention; do not upgrade PyTorch without checking or rebuilding that optional accelerated path.
- **Pedagogical Transparency**: Every mechanism (attention, causal masking, weight decay grouping, token packing, KV caching, tensor parallelism, serving queues, MoE routing) is written in readable, well-commented PyTorch.

---

## 2. Quickstart

Navigate to the cloned repository and install dependencies using `uv`:

```bash
# Clone the public companion repository
git clone https://github.com/brain-bzh/llm-course-companion.git
cd llm-course-companion

# Sync virtual environment with minimal dependencies
uv sync

# Run the test suite
uv run pytest tests/ -v
```

On a compatible Linux GPU environment (CPython 3.12, CUDA 13), install the
optional prebuilt FlashAttention path with:

```bash
uv sync --extra flash-attention
```

The extra uses the Torch version from the runtime environment while resolving
FlashAttention's build dependency, matching the setup used in the REVE project.

To run any module demo script:

```bash
uv run --extra gpt2 python scripts/01_gpt2_parity.py
uv run python scripts/02_overfit.py
uv run python scripts/09_kv_cache_bench.py
```

---

## 3. Module-by-Module Topic Coverage

| Module | Topic | Conceptual Coverage | Companion Code Module | Runnable Script | Verification / Milestone |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | [**Transformer from first principles**](https://brain-bzh.github.io/llm-course/companion/01-transformer/) | Reimplement explicit causal MHA, a Pre-LN Transformer block, embeddings, repeated blocks, and an LM head with tied weights. | [`starter/01_transformer.py`](starter/01_transformer.py)<br>[`nanolm/model.py`](nanolm/model.py)<br>[`nanolm/gpt2.py`](nanolm/gpt2.py) | [`scripts/01_gpt2_parity.py`](scripts/01_gpt2_parity.py) | **Exit criterion**: The student-built model reproduces official GPT-2 logits and a verified next-token log-probability. |
| **2** | **Training-loop anatomy & small-model overfit** | Shifted targets, AdamW parameter grouping, warmup/cosine decay, accumulation, norm clipping, and checkpoint recovery. | [`starter/02_training/`](starter/02_training/)<br>[`nanolm/optim.py`](nanolm/optim.py)<br>[`nanolm/train.py`](nanolm/train.py) | [`scripts/02_overfit.py`](scripts/02_overfit.py)<br>[`scripts/02_train_baseline.py`](scripts/02_train_baseline.py) | **Exit criterion**: A small NanoLM overfits one batch (`loss < 0.1`) and reloads with identical logits. |
| **3** | **Pretraining data pipeline** | Heuristic quality filtering, BPE tokenizer training, `<\|endoftext\|>` document delimiters, contiguous token sequence packing, binary uint16 memmap shards, and streaming batch loader. | [`nanolm/tokenizer.py`](nanolm/tokenizer.py)<br>[`nanolm/data.py`](nanolm/data.py) | [`scripts/03_prepare_dataset.py`](scripts/03_prepare_dataset.py) | Round-trip tokenization & memory-mapped slice validation. |
| **4** | **Single-GPU performance** | Static VRAM breakdown (weights, gradients, AdamW states), FLOPs counting per token, Model FLOPs Utilization (MFU), PyTorch SDPA, and mixed precision (`bfloat16`). | [`nanolm/profile_utils.py`](nanolm/profile_utils.py) | [`scripts/04_single_gpu_perf.py`](scripts/04_single_gpu_perf.py) | Measured tokens/second, MFU %, and memory footprint report. |
| **5** | **Distributed data parallelism** | Multi-GPU scaling with `torchrun`, `torch.distributed`, `DistributedDataParallel` wrapper, gradient all-reduce, accumulation with `no_sync()`, and global token accounting. | [`nanolm/distributed.py`](nanolm/distributed.py) | [`scripts/05_train_ddp.py`](scripts/05_train_ddp.py) | Verified synchronization, sample coverage, and global token counts. |
| **6** | **FSDP and ZeRO** | Parameter, gradient, and optimizer state sharding (ZeRO-1, ZeRO-2, ZeRO-3 / FSDP), PyTorch `FullyShardedDataParallel`, auto-wrap policies, and activation checkpointing. | [`nanolm/fsdp_utils.py`](nanolm/fsdp_utils.py) | [`scripts/06_fsdp_experiment.py`](scripts/06_fsdp_experiment.py) | Analytical and empirical comparison of peak memory vs DDP. |
| **7** | **Tensor parallelism** | Megatron-LM intra-layer model partitioning: ColumnParallelLinear (partition output dimension), RowParallelLinear (partition input dimension with all-reduce), and intermediate collective elimination. | [`nanolm/tensor_parallel.py`](nanolm/tensor_parallel.py) | [`scripts/07_tensor_parallel.py`](scripts/07_tensor_parallel.py) | Mathematical equivalence check between TP MLP and standard MLP. |
| **8** | **Context, pipeline & expert parallelism** | Context-length memory pressure, Ring Attention / Context Parallelism (CP), Pipeline Parallelism (PP) 1F1B schedule bubble overhead, and multidimensional cluster sizing. | [`nanolm/parallelism_calc.py`](nanolm/parallelism_calc.py) | [`scripts/08_parallelism_sizing.py`](scripts/08_parallelism_sizing.py) | Sizing report with VRAM, communication, and bubble estimates across cluster topologies. |
| **9** | **KV-cached decoding** | Autoregressive decoding bottlenecks ($O(T^2)$ compute vs $O(T)$), prefill vs decode stages, Key-Value cache tensor management, and MHA vs GQA cache footprint. | [`nanolm/kv_cache.py`](nanolm/kv_cache.py)<br>[`nanolm/model.py`](nanolm/model.py) | [`scripts/09_kv_cache_bench.py`](scripts/09_kv_cache_bench.py) | Strict token & logit equivalence test + latency speedup benchmark. |
| **10** | **Serving systems** | Serving constraints, continuous (iteration-level) batching scheduler, paged KV-cache memory allocation, Time To First Token (TTFT), Inter-Token Latency (ITL), and throughput. | [`nanolm/serving_sim.py`](nanolm/serving_sim.py) | [`scripts/10_serving_benchmark.py`](scripts/10_serving_benchmark.py) | Discrete-event serving simulation report with TTFT/ITL histograms. |
| **11** | **Frontier architectures** | Speculative decoding dynamics, DeepSeek MLA cache savings, linear attention, and Sparse Mixture of Experts (MoE) with top-k routing and load-balancing loss. | [`nanolm/moe.py`](nanolm/moe.py) | [`scripts/11_frontier_exploration.py`](scripts/11_frontier_exploration.py) | Speculative speedup modeling, MLA compression calculation, and MoE routing loss validation. |

---

## 4. Repository Structure

```text
llm-course-companion/
├── pyproject.toml              # Environment definition (torch, numpy, tiktoken, pytest)
├── README.md                   # Overview & module mapping guide
├── uv.lock                     # Locked dependencies for reproducible environments
├── starter/                    # Incomplete student starting points
│   ├── 01_transformer.py       # Module 1: MHA and Transformer TODO scaffold
│   └── 02_training/            # Module 2: optimizer and loop TODO scaffolds
├── nanolm/                     # Core library
│   ├── __init__.py             # Public exports (MiniGPT, GPTConfig, get_tokenizer)
│   ├── model.py                # Decoder-only GPT with causal attention & KV hooks
│   ├── gpt2.py                 # Provided Hugging Face GPT-2 weight converter
│   ├── optim.py                # AdamW weight decay grouping & cosine schedule
│   ├── tokenizer.py            # Tiktoken BPE wrapper with fallback
│   ├── data.py                 # Document packing, binary memmaps, batch loading
│   ├── train.py                # Training step, accumulation, clipping, checkpointing
│   ├── generate.py             # Autoregressive generation (cached & uncached)
│   ├── profile_utils.py        # FLOPs counter, MFU estimation, memory breakdown
│   ├── distributed.py          # DDP process group setup & all-reduce helpers
│   ├── fsdp_utils.py           # PyTorch FSDP auto-wrap policy and configuration
│   ├── tensor_parallel.py      # ColumnParallelLinear, RowParallelLinear, sharded MLP
│   ├── parallelism_calc.py     # Analytical 3D/4D/5D parallelism sizing calculator
│   ├── kv_cache.py             # Key-Value cache manager & benchmark
│   ├── serving_sim.py          # Continuous batching & paged memory simulation
│   └── moe.py                  # Sparse MoE layer with top-k router & aux loss
├── scripts/                    # Standalone executable module demonstrations
│   ├── 01_gpt2_parity.py       # Module 1: official GPT-2 logit parity
│   ├── 02_overfit.py           # Module 2: one-batch overfit & checkpoint round-trip
│   ├── 02_training_step.py     # Module 2: Optimizer step & checkpoint test
│   ├── 02_train_baseline.py    # Module 2: Baseline training & text sampling
│   ├── 03_prepare_dataset.py   # Module 3: BPE packing to binary memmap
│   ├── 04_single_gpu_perf.py   # Module 4: Memory breakdown, FLOPs & MFU
│   ├── 05_train_ddp.py         # Module 5: DDP distributed training launcher
│   ├── 06_fsdp_experiment.py   # Module 6: ZeRO-1/2/3 memory scaling comparison
│   ├── 07_tensor_parallel.py   # Module 7: TP column/row linear equivalence
│   ├── 08_parallelism_sizing.py# Module 8: Multidimensional cluster sizing & hierarchy
│   ├── 09_kv_cache_bench.py    # Module 9: KV-cache vs uncached benchmark
│   ├── 10_serving_benchmark.py # Module 10: Continuous batching simulation
│   └── 11_frontier_exploration.py # Module 11: Speculative decode, MLA, linear attention & MoE
└── tests/                      # Automated unit test suite
    ├── test_module_01_model.py
    ├── test_module_02_optim.py
    ├── test_module_03_data.py
    ├── test_module_07_tp.py
    ├── test_module_09_kv_cache.py
    └── test_module_11_frontier.py
```

---

## 5. Verification & Testing

Run all unit tests:

```bash
uv run pytest tests/ -v
```

Expected test outcomes:
- `test_module_01_model.py`: Progressively verifies explicit MHA, causality, the Transformer block, full-model shape, weight tying, and the GPT-2 mapping contract.
- `test_module_02_optim.py`: Verifies AdamW grouping, the learning-rate schedule, one-batch overfitting, and exact checkpoint recovery.
- `test_module_03_data.py`: Verifies BPE encoding/decoding and binary memmap batch alignment.
- `test_module_07_tp.py`: Verifies numerical equivalence between partitioned Column/Row linear layers and standard Linear layers.
- `test_module_09_kv_cache.py`: Verifies that KV-cached incremental generation produces tokens and logits identical to full uncached generation.
- `test_module_11_frontier.py`: Verifies MoE output shapes, top-k dispatch, and positive auxiliary load-balancing loss.
