"""Tests for Module 3 — BPE and the data pipeline."""

import tempfile
from minilm.tokenizer import BPETokenizer, get_tokenizer
from minilm.data import pack_documents, BinaryShardedDataset


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
