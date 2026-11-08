"""Session 10 starter — splits, packing, and shifted batches.

Copy this file to `nanolm/data.py` on your lab branch, implement the three
TODO functions in order, and run the matching tests after each one:

    uv run pytest tests/test_module_03_data.py -k split -q
    uv run pytest tests/test_module_03_data.py -k pack -q
    uv run pytest tests/test_module_03_data.py -k batch -q
    uv run python scripts/03_prepare_dataset.py

The filters, `normalize_text`, the memmap constructor, and the sequential
iterator are provided.
"""

import os
from typing import Dict, Generator, List, Tuple
import numpy as np
import torch


def filter_by_length(doc: str, min_chars: int = 50, max_chars: int = 100_000) -> bool:
    """Filter out documents that are too short (noise) or pathologically long."""
    length = len(doc.strip())
    return min_chars <= length <= max_chars


def filter_by_alpha_ratio(doc: str, min_ratio: float = 0.6) -> bool:
    """Filter out documents with too few alphabetic characters (e.g. junk binary or logs)."""
    if not doc:
        return False
    alphanumeric = sum(c.isalnum() or c.isspace() for c in doc)
    return (alphanumeric / len(doc)) >= min_ratio


def normalize_text(text: str) -> str:
    """Whitespace-normalized text used to detect exact duplicates."""
    return " ".join(text.split())


def assign_splits(records: List[Tuple[str, str, str]]) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]]]:
    """Deduplicate documents BEFORE splitting, so no text appears in two splits.

    `records` holds (doc_id, proposed_split, text) in a fixed order. Policy:
    the first record of each exact-duplicate group (after whitespace
    normalization) is kept in its proposed split; every later copy is dropped,
    wherever it was proposed.

    Returns ({split: [(doc_id, text), ...]}, [(dropped_id, kept_id), ...]).
    """
    # TODO: walk the records in order. Use normalize_text(text) as the duplicate key.
    #       Keep the first record of each key in its proposed split; record every
    #       later copy as (dropped_id, kept_id) and skip it.
    # Question: why must this happen BEFORE the tokenizer is trained?
    raise NotImplementedError


def pack_documents(
    documents: List[str],
    tokenizer,
    output_bin_path: str,
    eot_token_id: int,
    dtype: np.dtype = np.uint16,
) -> int:
    """Encode a list of text documents and write a contiguous packed binary file.

    Between each document, the end-of-text token ID is inserted.
    Returns the total number of tokens written.
    """
    # TODO: encode each document and append eot_token_id after EVERY document
    #       (including the last), producing one flat list of token IDs.
    # TODO: raise ValueError if any ID does not fit `dtype` (see np.iinfo).
    # TODO: write the IDs with np.asarray(ids, dtype=dtype).tofile(path) after
    #       creating the parent directory; return the number of tokens written.
    raise NotImplementedError


class BinaryShardedDataset:
    """Memory-mapped binary dataset reading contiguous token sequences."""

    def __init__(self, bin_path: str, dtype: np.dtype = np.uint16):
        self.bin_path = bin_path
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.num_tokens = len(self.data)

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous chunks of length `block_size`."""
        assert self.num_tokens > block_size, "Dataset too small for block_size"
        # TODO: draw `batch_size` random start offsets i such that the TARGET window
        #       data[i + 1 : i + 1 + block_size] still fits in the shard.
        # TODO: x = data[i : i + block_size], y = data[i + 1 : i + 1 + block_size],
        #       stacked as int64 tensors on `device`.
        # Question: what is the largest valid i, and what happens with one more?
        raise NotImplementedError

    def iterate_batches(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Deterministically iterate over the dataset sequentially."""
        seq_len = block_size
        step = batch_size * seq_len
        total_batches = (self.num_tokens - 1) // step
        for b in range(total_batches):
            start_idx = b * step
            x_chunks = []
            y_chunks = []
            for i in range(batch_size):
                offset = start_idx + i * seq_len
                x_chunks.append(self.data[offset : offset + seq_len].astype(np.int64))
                y_chunks.append(self.data[offset + 1 : offset + 1 + seq_len].astype(np.int64))
            x = torch.from_numpy(np.stack(x_chunks)).to(device)
            y = torch.from_numpy(np.stack(y_chunks)).to(device)
            yield x, y
