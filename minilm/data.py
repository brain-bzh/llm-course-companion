"""Data pipeline: document packing, binary memmaps, and batch loading.

Covers:
- Module 3: BPE and the data pipeline (packing, boundaries, memmap shards, batch loader).
"""

import os
from typing import List, Tuple, Generator
import numpy as np
import torch


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
    all_tokens: List[int] = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(eot_token_id)

    arr = np.array(all_tokens, dtype=dtype)
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    with open(output_bin_path, "wb") as f:
        f.write(arr.tobytes())

    return len(arr)


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
        ix = np.random.randint(0, self.num_tokens - block_size, size=batch_size)
        x_np = np.stack([self.data[i : i + block_size].astype(np.int64) for i in ix])
        y_np = np.stack([self.data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        return x, y

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
