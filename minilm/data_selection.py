"""Data selection, filtering, deduplication, and quality scoring.

Covers:
- Session 5: Data selection (filtering, deduplication, quality signals, controlled A/B subsets).
"""

import hashlib
import re
from typing import List, Set, Tuple, Dict, Any
import numpy as np


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


def exact_deduplicate(documents: List[str]) -> Tuple[List[str], int]:
    """Exact deduplication via SHA-256 hashing.

    Returns the unique documents and count of duplicates removed.
    """
    seen_hashes: Set[str] = set()
    unique_docs: List[str] = []
    duplicates_count = 0

    for doc in documents:
        # Normalize whitespace
        norm = " ".join(doc.strip().split())
        doc_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)
        else:
            duplicates_count += 1

    return unique_docs, duplicates_count


def minhash_deduplicate(
    documents: List[str],
    n_gram: int = 5,
    threshold: float = 0.8,
) -> Tuple[List[str], int]:
    """Simple n-gram Jaccard deduplication.

    Identifies and discards documents sharing > threshold n-grams with prior documents.
    """
    def get_shingles(text: str) -> Set[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < n_gram:
            return set(words)
        return {" ".join(words[i : i + n_gram]) for i in range(len(words) - n_gram + 1)}

    seen_shingle_sets: List[Set[str]] = []
    selected: List[str] = []
    removed = 0

    for doc in documents:
        shingles = get_shingles(doc)
        if not shingles:
            continue
        is_duplicate = False
        for seen in seen_shingle_sets:
            intersection = len(shingles & seen)
            union = len(shingles | seen)
            if union > 0 and (intersection / union) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            seen_shingle_sets.append(shingles)
            selected.append(doc)
        else:
            removed += 1

    return selected, removed


def create_controlled_subsets(
    raw_documents: List[str],
    target_token_count: int,
    tokenizer,
) -> Dict[str, List[str]]:
    """Construct random baseline and curated/selected subsets with matched token budget."""
    # 1. Curated pipeline: filter + dedup
    filtered = [d for d in raw_documents if filter_by_length(d) and filter_by_alpha_ratio(d)]
    curated_pool, _ = exact_deduplicate(filtered)

    # Accumulate tokens up to target_token_count for both
    def gather_to_budget(pool: List[str]) -> List[str]:
        budget_docs = []
        tokens_so_far = 0
        for doc in pool:
            count = len(tokenizer.encode(doc))
            if tokens_so_far + count > target_token_count and budget_docs:
                break
            budget_docs.append(doc)
            tokens_so_far += count
        return budget_docs

    # Random baseline (shuffled raw docs)
    rng = np.random.default_rng(42)
    raw_shuffled = raw_documents.copy()
    rng.shuffle(raw_shuffled)
    random_subset = gather_to_budget(raw_shuffled)

    # Curated subset
    curated_subset = gather_to_budget(curated_pool)

    return {
        "random_baseline": random_subset,
        "selected_curated": curated_subset,
    }
