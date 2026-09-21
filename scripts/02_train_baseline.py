"""Module 2 — Training-loop anatomy and baseline GPT.

Demonstrates:
- Baseline training run with healthy validation loss tracking.
- Autoregressive text sampling as qualitative diagnostic.
"""

import tempfile
import torch
from minilm.model import MiniGPT, GPTConfig
from minilm.tokenizer import get_tokenizer
from minilm.data import pack_documents, BinaryShardedDataset
from minilm.optim import configure_optimizers
from minilm.train import train_step, optimizer_step, evaluate_loss
from minilm.generate import generate_uncached

def main():
    print("=== Module 2: Train Baseline Small GPT ===")
    tokenizer = get_tokenizer()

    # Small synthetic corpus of repetitive rhythmic text for fast learning
    corpus = [
        "one two three four five six seven eight nine ten. " * 20,
        "alpha beta gamma delta epsilon zeta eta theta. " * 20,
    ] * 5

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = tmp.name

    pack_documents(corpus, tokenizer, bin_path, tokenizer.eot_token_id)
    dataset = BinaryShardedDataset(bin_path)

    config = GPTConfig(
        vocab_size=getattr(tokenizer, "n_words", 50257),
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = MiniGPT(config)
    optimizer = configure_optimizers(model, learning_rate=2e-3)

    print(f"Training Small GPT ({model.get_num_params():,} params) for 60 steps...")
    for step in range(60):
        x, y = dataset.get_batch(batch_size=4, block_size=config.block_size)
        loss = train_step(model, optimizer, x, y)
        optimizer_step(model, optimizer)

        if (step + 1) % 20 == 0:
            val_loss = evaluate_loss(model, dataset, eval_iters=5, batch_size=4, block_size=config.block_size)
            print(f"  Step {step+1:02d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

    print("\nQualitative diagnostic: Sampling from trained baseline:")
    prompt = "one two"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    sampled = generate_uncached(model, prompt_ids, max_new_tokens=15, temperature=0.7)
    decoded = tokenizer.decode(sampled[0].tolist())
    print(f"Prompt: {prompt!r} -> Generated: {decoded!r}")

if __name__ == "__main__":
    main()
