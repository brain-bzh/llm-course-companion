"""Session 5 — Data selection.

Demonstrates:
- Heuristic filtering (length, alphanumeric ratio).
- Exact and near deduplication.
- Controlled A/B subset generation under fixed token budget.
"""

from minilm.tokenizer import get_tokenizer
from minilm.data_selection import (
    filter_by_length,
    filter_by_alpha_ratio,
    exact_deduplicate,
    minhash_deduplicate,
    create_controlled_subsets,
)

def main():
    print("=== Session 5: Data Selection & Controlled Subsets ===")
    tokenizer = get_tokenizer()

    raw_corpus = [
        "Too short",
        "%%% $$$ !!! #### @@@@ &&&& ????",  # Low alpha ratio
        "The quick brown fox jumps over the lazy dog. A classic sentence for testing language modeling pipelines.",
        "The quick brown fox jumps over the lazy dog. A classic sentence for testing language modeling pipelines.",  # Exact dup
        "The fast brown fox jumps over the lazy dog. A classic sentence for testing language modeling pipelines.",   # Near dup
        "Distributed data parallelism replicates model weights on every GPU while synchronizing gradients with all-reduce.",
        "Deep learning models require balanced curricula and rigorous data filtering to avoid wasting compute.",
    ]

    print(f"Raw corpus size: {len(raw_corpus)} documents")

    # 1. Filter checks
    valid_len = [d for d in raw_corpus if filter_by_length(d, min_chars=30)]
    print(f"Passed length filter (>=30 chars): {len(valid_len)}")

    valid_alpha = [d for d in valid_len if filter_by_alpha_ratio(d, min_ratio=0.7)]
    print(f"Passed alphanumeric ratio filter (>=0.7): {len(valid_alpha)}")

    # 2. Exact Deduplication
    exact_unique, num_exact_dups = exact_deduplicate(valid_alpha)
    print(f"Removed {num_exact_dups} exact duplicates. Remaining: {len(exact_unique)}")

    # 3. Controlled Subsets (Fixed token budget)
    subsets = create_controlled_subsets(raw_corpus, target_token_count=50, tokenizer=tokenizer)
    print("\nControlled Experiment Subsets:")
    print(f"  Random Baseline Subset: {len(subsets['random_baseline'])} docs")
    print(f"  Curated Selected Subset: {len(subsets['selected_curated'])} docs")

if __name__ == "__main__":
    main()
