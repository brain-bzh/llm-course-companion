"""Session 3 — BPE and the data pipeline.

Demonstrates:
- Training an educational Byte-level BPE tokenizer from scratch.
- Inspecting learned subword merges and compression ratios.
- Document boundary handling (<|endoftext|>).
- Binary uint16 packing to disk.
- Fast memory-mapped batch loading and shifted target verification.
"""

import tempfile
from minilm.tokenizer import BPETokenizer
from minilm.data import pack_documents, BinaryShardedDataset


def main():
    print("=== Session 3: BPE Tokenizer and Packed Data Pipeline ===")

    # 1. Raw corpus for training the BPE tokenizer
    sample_corpus = """
    Language models are trained on large text corpora using self-supervised learning.
    The Transformer architecture relies on multi-head scaled dot-product attention.
    Continuous batching and paged memory increase serving throughput under high load.
    Byte-pair encoding builds a vocabulary of subword units by iteratively merging frequent pairs.
    Through subword tokenization, rare words decompose into meaningful subunits, avoiding out-of-vocabulary tokens.
    """

    print("\n--- Step 1: Train BPE Tokenizer From Scratch ---")
    tokenizer = BPETokenizer()
    target_vocab_size = 320  # 256 base bytes + 1 special token + 63 merges
    tokenizer.train(sample_corpus, vocab_size=target_vocab_size)
    print(f"BPE training complete! Base bytes: 256 | Merges: {len(tokenizer.merges)} | Total vocab: {tokenizer.vocab_size}")
    print(f"EOT token ID: {tokenizer.eot_token_id}")

    # Inspect top 5 learned merges
    print("\nSample learned merges (pair IDs -> merged token):")
    for idx, (pair, new_id) in enumerate(list(tokenizer.merges.items())[:5]):
        p0_repr = tokenizer.vocab[pair[0]].decode("utf-8", errors="replace")
        p1_repr = tokenizer.vocab[pair[1]].decode("utf-8", errors="replace")
        merged_repr = tokenizer.vocab[new_id].decode("utf-8", errors="replace")
        print(f"  Merge {idx + 1}: ('{p0_repr}', '{p1_repr}') -> '{merged_repr}' (Token ID: {new_id})")

    # Measure compression ratio
    test_sentence = "Language models use byte-pair encoding for subword tokenization."
    raw_bytes = len(test_sentence.encode("utf-8"))
    token_ids = tokenizer.encode(test_sentence)
    compression = raw_bytes / len(token_ids)
    print(f"\nCompression metric for: \"{test_sentence}\"")
    print(f"  Raw bytes: {raw_bytes} | Tokens: {len(token_ids)} | Compression ratio: {compression:.2f}x bytes/token")

    # Verify round-trip decoding
    decoded = tokenizer.decode(token_ids)
    assert decoded == test_sentence, "Decoded text does not match original!"
    print("  Round-trip decode check: PASSED")

    # 2. Document packing with <|endoftext|>
    print("\n--- Step 2: Document Packing and Binary Sharding ---")
    documents = [
        "First document: Pretraining trains on next-token prediction across vast text collections.",
        "Second document: Scaled dot-product attention computes interactions between queries and keys.",
        "Third document: Rotary position embeddings inject positional information into attention projections.",
    ]

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name

    total_tokens = pack_documents(documents, tokenizer, bin_path, tokenizer.eot_token_id)
    print(f"Packed {len(documents)} documents into binary shard: {bin_path}")
    print(f"Total tokens written: {total_tokens} (uint16)")

    # 3. Fast memory-mapped loading
    print("\n--- Step 3: Fast Memory-Mapped Batch Loading ---")
    dataset = BinaryShardedDataset(bin_path)
    print(f"Dataset memory-mapped successfully. Shard token count: {dataset.num_tokens}")

    # Sample a batch of (X, Y)
    batch_size = 2
    block_size = 16
    x, y = dataset.get_batch(batch_size=batch_size, block_size=block_size)
    print(f"Sampled batch shapes -> Input X: {list(x.shape)}, Target Y: {list(y.shape)}")

    # Verify target offset invariant (Y is shifted by 1 relative to X)
    is_shifted = (y[:, :-1] == x[:, 1:]).all().item()
    print(f"Target shift invariant verified (Y[:, :-1] == X[:, 1:]): {is_shifted}")
    assert is_shifted, "Target offset invariant failed!"

    print("\nSUCCESS: Session 3 data pipeline verified!")


if __name__ == "__main__":
    main()
