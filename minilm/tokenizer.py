"""Tokenizer implementations: Byte-Pair Encoding (BPE), Tiktoken, and Char fallback.

Covers:
- Session 3: BPE and the data pipeline (vocabulary, byte-level BPE, special tokens, encoding/decoding).
"""

from collections import Counter
import re
from typing import List, Dict, Tuple, Optional, Set


class BPETokenizer:
    """Educational Byte-level Byte-Pair Encoding (BPE) Tokenizer.

    Grounded in:
    - Philip Gage (1994): 'A New Algorithm for Data Compression'
    - Radford et al. (2019): 'Language Models are Unsupervised Multitask Learners' (GPT-2)
    - Sebastian Raschka (2025): 'Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch'
    """

    def __init__(self):
        # 0..255 are base byte tokens (guarantees 0 out-of-vocabulary tokens)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special: Dict[int, str] = {}

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Train BPE merge rules on raw text up to the requested vocab_size."""
        assert vocab_size >= 256, "vocab_size must be at least 256 to cover base bytes"
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]

        # Register special tokens immediately above 255
        next_id = 256
        for tok in special_tokens:
            self.special_tokens[tok] = next_id
            self.inverse_special[next_id] = tok
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        # Convert training text to initial byte token IDs
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)

        # Iteratively find and merge the most frequent adjacent pair
        num_merges = vocab_size - next_id
        for _ in range(num_merges):
            if len(ids) < 2:
                break
            pairs = Counter(zip(ids[:-1], ids[1:]))
            if not pairs:
                break
            best_pair, best_count = pairs.most_common(1)[0]
            if best_count <= 1:
                break  # No more repeated pairs to merge

            new_id = next_id
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            next_id += 1

            # In-place pair replacement
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

    @property
    def eot_token_id(self) -> int:
        return self.special_tokens.get("<|endoftext|>", 256)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """Encode text into token IDs, resolving special tokens and learned BPE merges."""
        if allowed_special is None:
            allowed_special = set(self.special_tokens.keys())

        # Split string around allowed special tokens
        if allowed_special:
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        tokens: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in allowed_special:
                tokens.append(self.special_tokens[part])
                continue

            # Convert text segment to raw bytes
            ids = list(part.encode("utf-8"))
            while len(ids) >= 2:
                pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
                # Find earliest learned merge rule (lowest merge ID)
                mergeable = [(self.merges[p], p) for p in pairs if p in self.merges]
                if not mergeable:
                    break
                best_new_id, best_pair = min(mergeable, key=lambda x: x[0])

                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                        new_ids.append(best_new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            tokens.extend(ids)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back into a UTF-8 string."""
        byte_chunks = []
        for tok in tokens:
            if tok in self.inverse_special:
                byte_chunks.append(self.inverse_special[tok].encode("utf-8"))
            elif tok in self.vocab:
                byte_chunks.append(self.vocab[tok])
            else:
                byte_chunks.append(b"")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")


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

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

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

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

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
