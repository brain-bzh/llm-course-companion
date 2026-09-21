# MiniLM Companion Code Repository

Companion codebase for the course **"Training and Scaling Language Models: From First Principles to Efficient Serving"**, designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).

This repository provides an ultra-lean, pedagogical implementation of an end-to-end language modeling system. It is designed to be completely inspectable without heavy third-party framework abstractions.

---

## 1. Design & Dependency Philosophy

- **Minimal Dependencies**: The entire codebase runs with only **three runtime dependencies**:
  - `torch` (Tensor operations, automatic differentiation, neural network primitives)
  - `numpy` (Binary memory mapping and numerical helpers)
  - `tiktoken` (Standard Byte-Pair Encoding tokenizer for GPT-style models)
  - *(`pytest` for test verification)*
- **Self-Contained & Isolated**: Lives in `companion/` with its own `pyproject.toml`, keeping the root course website environment lightweight and fast.
- **Pedagogical Transparency**: Every mechanism (attention, causal masking, weight decay grouping, token packing, KV caching, tensor parallelism, serving queues, MoE routing) is written in readable, well-commented PyTorch.

---

## 2. Quickstart

Navigate to the `companion/` directory and install dependencies using `uv`:

```bash
cd companion

# Sync virtual environment with minimal dependencies
uv sync

# Run the test suite
uv run pytest tests/ -v
```

To run any module demo script:

```bash
uv run python scripts/01_overfit.py
uv run python scripts/11_kv_cache_bench.py
```

---

## 3. Module-by-Module Topic Coverage

| Module | Topic | Conceptual Coverage | Companion Code Module | Runnable Script | Verification / Milestone |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Transformer from first principles** | Token & position embeddings, scaled dot-product attention, causal masking, Multi-Head Attention, Pre-LN residual connections, MLP, and LM head with tied weights. | [`minilm/model.py`](minilm/model.py) | [`scripts/01_overfit.py`](scripts/01_overfit.py) | **Exit criterion**: Tiny-batch overfit (`loss < 0.1`) & causal masking invariance. |
| **2** | **Training-loop anatomy** | Cross-entropy loss, AdamW decoupled weight decay (2D vs 1D parameters), cosine decay with linear warmup, gradient accumulation, norm clipping, and checkpoint save/recovery. | [`minilm/optim.py`](minilm/optim.py)<br>[`minilm/train.py`](minilm/train.py) | [`scripts/02_training_step.py`](scripts/02_training_step.py) | Optimizer parameter grouping test & checkpoint round-trip. |
| **3** | **BPE and the data pipeline** | Byte-Pair Encoding (BPE), `<\|endoftext\|>` document delimiters, contiguous token sequence packing, binary uint16 memmap shards, and streaming batch loader. | [`minilm/tokenizer.py`](minilm/tokenizer.py)<br>[`minilm/data.py`](minilm/data.py) | [`scripts/03_prepare_dataset.py`](scripts/03_prepare_dataset.py) | Round-trip tokenization & memory-mapped slice validation. |
| **4** | **Train a small GPT** | Full training run execution on prepared dataset, validation loss curves, checkpoint selection, and autoregressive sampling (temperature, top-k/top-p) as a qualitative diagnostic. | [`minilm/train.py`](minilm/train.py)<br>[`minilm/generate.py`](minilm/generate.py) | [`scripts/04_train_baseline.py`](scripts/04_train_baseline.py) | Baseline checkpoint & generated text sample inspection. |
| **5** | **Data selection** | Quality filtering (length, alphanumeric ratio), exact hashing & near-duplicate (MinHash/shingling) deduplication, and controlled A/B subsets under fixed token budgets. | [`minilm/data_selection.py`](minilm/data_selection.py) | [`scripts/05_data_selection.py`](scripts/05_data_selection.py) | Controlled comparison of random vs curated corpus. |
| **6** | **Single-GPU performance** | Static VRAM breakdown (weights, gradients, AdamW states), FLOPs counting per token, Model FLOPs Utilization (MFU), PyTorch SDPA, and mixed precision (`bfloat16`). | [`minilm/profile_utils.py`](minilm/profile_utils.py) | [`scripts/06_single_gpu_perf.py`](scripts/06_single_gpu_perf.py) | Measured tokens/second, MFU %, and memory footprint report. |
| **7** | **Distributed data parallelism** | Multi-GPU scaling with `torchrun`, `torch.distributed`, `DistributedDataParallel` wrapper, gradient all-reduce, accumulation with `no_sync()`, and global token accounting. | [`minilm/distributed.py`](minilm/distributed.py) | [`scripts/07_train_ddp.py`](scripts/07_train_ddp.py) | Verified synchronization, sample coverage, and global token counts. |
| **8** | **FSDP and ZeRO** | Parameter, gradient, and optimizer state sharding (ZeRO-1, ZeRO-2, ZeRO-3 / FSDP), PyTorch `FullyShardedDataParallel`, auto-wrap policies, and activation checkpointing. | [`minilm/fsdp_utils.py`](minilm/fsdp_utils.py) | [`scripts/08_fsdp_experiment.py`](scripts/08_fsdp_experiment.py) | Analytical and empirical comparison of peak memory vs DDP. |
| **9** | **Tensor parallelism** | Megatron-LM intra-layer model partitioning: ColumnParallelLinear (partition output dimension), RowParallelLinear (partition input dimension with all-reduce), and intermediate collective elimination. | [`minilm/tensor_parallel.py`](minilm/tensor_parallel.py) | [`scripts/09_tensor_parallel.py`](scripts/09_tensor_parallel.py) | Mathematical equivalence check between TP MLP and standard MLP. |
| **10** | **Context and pipeline parallelism** | Context-length memory pressure, Ring Attention / Context Parallelism (CP), Pipeline Parallelism (PP) 1F1B schedule bubble overhead, and 3D/4D cluster sizing. | [`minilm/parallelism_calc.py`](minilm/parallelism_calc.py) | [`scripts/10_parallelism_sizing.py`](scripts/10_parallelism_sizing.py) | Sizing report with VRAM, communication, and bubble estimates across cluster topologies. |
| **11** | **KV-cached decoding** | Autoregressive decoding bottlenecks ($O(T^2)$ compute vs $O(T)$), prefill vs decode stages, Key-Value cache tensor management, and MHA vs GQA cache footprint. | [`minilm/kv_cache.py`](minilm/kv_cache.py)<br>[`minilm/model.py`](minilm/model.py) | [`scripts/11_kv_cache_bench.py`](scripts/11_kv_cache_bench.py) | Strict token & logit equivalence test + latency speedup benchmark. |
| **12** | **Serving systems** | Serving constraints, continuous (iteration-level) batching scheduler, paged KV-cache memory allocation, Time To First Token (TTFT), Inter-Token Latency (ITL), and throughput. | [`minilm/serving_sim.py`](minilm/serving_sim.py) | [`scripts/12_serving_benchmark.py`](scripts/12_serving_benchmark.py) | Discrete-event serving simulation report with TTFT/ITL histograms. |
| **13** | **Beyond dense Transformers** | Sparse Mixture of Experts (MoE), top-k routing, auxiliary load-balancing loss, active compute ratio vs parameter capacity, and state-space model trade-offs. | [`minilm/moe.py`](minilm/moe.py) | [`scripts/13_moe_exploration.py`](scripts/13_moe_exploration.py) | MoE capacity vs compute ratio analysis and routing loss validation. |

---

## 4. Repository Structure

```text
companion/
├── pyproject.toml              # Environment definition (torch, numpy, tiktoken, pytest)
├── README.md                   # This overview & module mapping guide
├── minilm/                     # Core library
│   ├── __init__.py             # Public exports (MiniGPT, GPTConfig, get_tokenizer)
│   ├── model.py                # Decoder-only GPT with causal attention & KV hooks
│   ├── optim.py                # AdamW weight decay grouping & cosine schedule
│   ├── tokenizer.py            # Tiktoken BPE wrapper with fallback
│   ├── data.py                 # Document packing, binary memmaps, batch loading
│   ├── train.py                # Training step, accumulation, clipping, checkpointing
│   ├── generate.py             # Autoregressive generation (cached & uncached)
│   ├── data_selection.py       # Heuristic filtering, deduplication, A/B subsets
│   ├── profile_utils.py        # FLOPs counter, MFU estimation, memory breakdown
│   ├── distributed.py          # DDP process group setup & all-reduce helpers
│   ├── fsdp_utils.py           # PyTorch FSDP auto-wrap policy and configuration
│   ├── tensor_parallel.py      # ColumnParallelLinear, RowParallelLinear, sharded MLP
│   ├── parallelism_calc.py     # Analytical 3D/4D parallelism sizing calculator
│   ├── kv_cache.py             # Key-Value cache manager & benchmark
│   ├── serving_sim.py          # Continuous batching & paged memory simulation
│   └── moe.py                  # Sparse MoE layer with top-k router & aux loss
├── scripts/                    # Standalone executable module demonstrations
│   ├── 01_overfit.py           # Module 1: Tiny-batch overfit test
│   ├── 02_training_step.py     # Module 2: Optimizer step & checkpoint test
│   ├── 02_train_baseline.py    # Module 2: Baseline training & text sampling
│   ├── 03_prepare_dataset.py   # Module 3: BPE packing to binary memmap
│   ├── 04_data_selection.py    # Module 4: Quality filter & deduplication demo
│   ├── 05_single_gpu_perf.py   # Module 5: Memory breakdown, FLOPs & MFU
│   ├── 06_train_ddp.py         # Module 6: DDP distributed training launcher
│   ├── 07_fsdp_experiment.py   # Module 7: ZeRO-1/2/3 memory scaling comparison
│   ├── 08_tensor_parallel.py   # Module 8: TP column/row linear equivalence
│   ├── 09_parallelism_sizing.py# Module 9: Multidimensional cluster sizing & hierarchy
│   ├── 10_kv_cache_bench.py    # Module 10: KV-cache vs uncached benchmark
│   ├── 11_serving_benchmark.py # Module 11: Continuous batching simulation
│   └── 12_frontier_exploration.py # Module 12: Speculative decode, MLA, linear attention & MoE
└── tests/                      # Automated unit test suite
    ├── test_module_01_model.py
    ├── test_module_02_optim.py
    ├── test_module_03_data.py
    ├── test_module_08_tp.py
    ├── test_module_10_kv_cache.py
    └── test_module_12_frontier.py
```

---

## 5. Verification & Testing

Run all unit tests:

```bash
cd companion
uv run pytest -v
```

Expected test outcomes:
- `test_module_01_model.py`: Verifies output shape, causal masking invariance (future tokens cannot affect past logits), and tiny-batch overfit.
- `test_module_02_optim.py`: Verifies weight decay applies only to $\ge 2\text{D}$ tensors, and verifies warmup/cosine decay schedules.
- `test_module_03_data.py`: Verifies BPE encoding/decoding and binary memmap batch alignment.
- `test_module_09_tp.py`: Verifies numerical equivalence between partitioned Column/Row linear layers and standard Linear layers.
- `test_module_11_kv_cache.py`: Verifies that KV-cached incremental generation produces tokens and logits identical to full uncached generation.
- `test_module_13_moe.py`: Verifies MoE output shapes, top-k dispatch, and positive auxiliary load-balancing loss.
