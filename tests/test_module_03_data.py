"""Tests for Module 3 — BPE and the data pipeline."""

import tempfile
from nanolm.tokenizer import BPETokenizer, get_tokenizer
from nanolm.data import pack_documents, BinaryShardedDataset


def test_bpe_tokenizer_train_and_round_trip():
    """Verify BPE training from scratch, learned merges, and lossless decoding."""
    tokenizer = BPETokenizer()
    corpus = (
        "low lower lowest newest widest "
        "subword tokenization enables vocabulary compression without out-of-vocabulary tokens."
    )
    tokenizer.train(corpus, vocab_size=280)

    # Base 256 + 1 special + at least some merges
    assert tokenizer.vocab_size <= 280
    assert len(tokenizer.merges) > 0
    assert tokenizer.eot_token_id == 256

    # Test round trip with standard text and special token
    test_text = "lowest subword tokenization <|endoftext|> unseen utf-8: café 🚀"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text


def test_bpe_compression():
    """Verify that trained BPE produces fewer tokens than raw byte length."""
    tokenizer = BPETokenizer()
    corpus = "Transformer architectures rely on multi-head scaled dot-product self-attention mechanisms. " * 8
    tokenizer.train(corpus, vocab_size=320)

    test_sentence = "Transformer self-attention mechanisms."
    raw_bytes = len(test_sentence.encode("utf-8"))
    tokens = tokenizer.encode(test_sentence)

    # Tokens must compress the byte representation
    assert len(tokens) < raw_bytes


def test_tokenizer_factory_round_trip():
    """Verify default tokenizer factory round-trip."""
    tokenizer = get_tokenizer()
    text = "Attention is all you need for scalable deep learning."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert text == decoded or text in decoded


def test_packed_dataset_batching_and_target_shift():
    """Verify document packing with EOT, binary memmap reading, and shifted targets."""
    tokenizer = BPETokenizer()
    corpus = "First document content. Second document content is longer. Third document."
    tokenizer.train(corpus, vocab_size=275)

    docs = [
        "First document content.",
        "Second document content is longer.",
        "Third document ends here.",
    ]

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name

    total_tokens = pack_documents(docs, tokenizer, bin_path, tokenizer.eot_token_id)
    dataset = BinaryShardedDataset(bin_path)

    assert dataset.num_tokens == total_tokens

    # Verify batch shape and target offset
    batch_size = 2
    block_size = 6
    x, y = dataset.get_batch(batch_size=batch_size, block_size=block_size)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # Next-token prediction invariant: y at position t is x at position t+1
    assert (y[:, :-1] == x[:, 1:]).all().item()


def test_assign_splits_removes_cross_split_duplicate():
    from nanolm.data import assign_splits

    records = [
        ("train-1", "train", "A small model learns from a sequence of tokens."),
        ("train-2", "train", "A token sequence can cross a document boundary."),
        ("val-1", "validation", "A held-out document measures a different example."),
        ("val-2", "validation", "A small  model learns from a sequence of tokens.\n"),
    ]
    splits, dropped = assign_splits(records)
    assert [d for d, _ in splits["train"]] == ["train-1", "train-2"]
    assert [d for d, _ in splits["validation"]] == ["val-1"]
    assert dropped == [("val-2", "train-1")]
    texts = {t for docs in splits.values() for _, t in docs}
    assert len(texts) == 3


def test_pack_places_eot_after_every_document(tmp_path):
    import numpy as np
    from nanolm.tokenizer import SimpleCharTokenizer

    tokenizer = SimpleCharTokenizer()
    docs = ["ab", "c", "de"]
    n = pack_documents(docs, tokenizer, str(tmp_path / "shard.bin"), tokenizer.eot_token_id)
    ids = np.fromfile(tmp_path / "shard.bin", dtype=np.uint16).tolist()
    assert n == len(ids) == 2 + 1 + 1 + 1 + 2 + 1
    eot_positions = [i for i, t in enumerate(ids) if t == tokenizer.eot_token_id]
    assert eot_positions == [2, 4, 7]
    assert tokenizer.decode(ids[:2]) == "ab"


def test_pack_rejects_ids_that_overflow_dtype(tmp_path):
    import numpy as np
    import pytest

    class Wide:
        def encode(self, text):
            return [70_000]

    with pytest.raises(ValueError):
        pack_documents(["x"], Wide(), str(tmp_path / "bad.bin"), 0, dtype=np.uint16)


def test_get_batch_offsets_stay_inside_shard_and_targets_shift(tmp_path):
    import numpy as np

    n, block = 20, 4
    np.arange(n, dtype=np.uint16).tofile(tmp_path / "range.bin")
    dataset = BinaryShardedDataset(str(tmp_path / "range.bin"))
    np.random.seed(0)
    starts = set()
    for _ in range(200):
        x, y = dataset.get_batch(batch_size=4, block_size=block)
        assert x.dtype == y.dtype
        assert (y == x + 1).all()  # on 0..n-1 data the target is the next value
        starts.update(x[:, 0].tolist())
    assert max(starts) == n - block - 1  # last window whose target still fits
    assert min(starts) == 0
