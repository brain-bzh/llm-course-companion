"""Tokenizer wrapper for Byte-Pair Encoding (BPE).

Covers:
- Session 3: BPE and the data pipeline (vocabulary, special tokens, encoding/decoding).
"""

from typing import List, Optional


class SimpleCharTokenizer:
    """Fallback character-level tokenizer for zero-dependency testing without network."""

    def __init__(self, vocab: Optional[str] = None):
        if vocab is None:
            # Printable ASCII + newline
            self.chars = sorted(list(set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;\"'-\n")))
        else:
            self.chars = sorted(list(set(vocab)))
        self.eot_token = "<|endoftext|>"
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.eot_id = len(self.chars)
        self.stoi[self.eot_token] = self.eot_id
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.n_words = len(self.stoi)

    @property
    def eot_token_id(self) -> int:
        return self.eot_id

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        tokens = []
        parts = text.split(self.eot_token)
        for idx, part in enumerate(parts):
            for ch in part:
                tokens.append(self.stoi.get(ch, 0))
            if idx < len(parts) - 1:
                tokens.append(self.eot_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos.get(t, "") for t in tokens])


class TiktokenTokenizer:
    """Wrapper around openai/tiktoken for GPT-2 / GPT-4 BPE encodings."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot_id = self.enc.eot_token

    @property
    def eot_token_id(self) -> int:
        return self.eot_id

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)


def get_tokenizer(name: str = "gpt2"):
    """Factory to retrieve tiktoken or fallback tokenizer."""
    try:
        import tiktoken  # noqa: F401
        return TiktokenTokenizer(name)
    except Exception:
        # Fallback to local character tokenizer
        return SimpleCharTokenizer()
