"""Session 3 — BPE and the data pipeline.

Demonstrates:
- BPE tokenization.
- Document boundary handling (<|endoftext|>).
- Binary uint16 packing to disk.
- Fast memory-mapped batch loading.
"""

import tempfile
from minilm.tokenizer import get_tokenizer
from minilm.data import pack_documents, BinaryShardedDataset

def main():
    print("=== Session 3: Tokenizer and Packed Data Pipeline ===")
    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded. EOT Token ID: {tokenizer.eot_token_id}")

    sample_docs = [
        "Language models are trained on large text corpora using self-supervised learning.",
        "The Transformer architecture relies on multi-head scaled dot-product attention.",
        "Continuous batching and paged memory increase serving throughput under high load.",
    ]

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name

    total_tokens = pack_documents(sample_docs, tokenizer, bin_path, tokenizer.eot_token_id)
    print(f"Packed {len(sample_docs)} documents into {bin_path}")
    print(f"Total tokens written: {total_tokens}")

    # Load dataset with memmap
    dataset = BinaryShardedDataset(bin_path)
    print(f"Dataset memory-mapped successfully. Shard token count: {dataset.num_tokens}")

    # Inspect a tiny batch
    x, y = dataset.get_batch(batch_size=2, block_size=8)
    print(f"Sampled batch shapes -> X: {list(x.shape)}, Y: {list(y.shape)}")
    print(f"Target Y is shifted by 1 relative to X: {(y[0, :-1] == x[0, 1:]).all().item()}")

if __name__ == "__main__":
    main()
