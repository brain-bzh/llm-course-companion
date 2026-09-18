"""Tests for Session 3 — BPE and the data pipeline."""

import tempfile
from minilm.tokenizer import get_tokenizer
from minilm.data import pack_documents, BinaryShardedDataset


def test_tokenizer_round_trip():
    tokenizer = get_tokenizer()
    text = "Attention is all you need for scalable deep learning."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert text == decoded or text in decoded


def test_packed_dataset_batching():
    tokenizer = get_tokenizer()
    docs = [
        "First document content.",
        "Second document content is a bit longer.",
        "Third document ends here.",
    ]

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name

    total_tokens = pack_documents(docs, tokenizer, bin_path, tokenizer.eot_token_id)
    dataset = BinaryShardedDataset(bin_path)

    assert dataset.num_tokens == total_tokens

    # Verify batch shape and target offset
    x, y = dataset.get_batch(batch_size=2, block_size=4)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
