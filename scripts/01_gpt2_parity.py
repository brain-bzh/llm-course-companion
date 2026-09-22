"""Session 1 capstone: reproduce GPT-2 logits with the student-built model."""

import torch
import torch.nn.functional as F

from nanolm.gpt2 import load_pretrained_gpt2
from nanolm.tokenizer import get_tokenizer


PROMPT = "The capital of Germany is Berlin. The capital of France is"
CONTINUATION = " Paris"
EXPECTED_LOG_PROB = -0.334223


def main() -> None:
    print("=== Session 1: GPT-2 parity ===")
    print("Loading the official GPT-2 checkpoint (downloaded once, then cached)...")
    model, reference = load_pretrained_gpt2("gpt2")
    tokenizer = get_tokenizer("gpt2")

    input_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)
    continuation_ids = tokenizer.encode(CONTINUATION)
    if len(continuation_ids) != 1:
        raise RuntimeError(f"Expected one GPT-2 token for {CONTINUATION!r}, got {continuation_ids}")

    with torch.no_grad():
        nanolm_logits, _, _ = model(input_ids)
        reference_logits = reference(input_ids).logits

    torch.testing.assert_close(nanolm_logits, reference_logits, rtol=1e-4, atol=1e-4)
    max_difference = (nanolm_logits - reference_logits).abs().max().item()

    next_log_probs = F.log_softmax(nanolm_logits[0, -1], dim=-1)
    continuation_id = continuation_ids[0]
    continuation_log_prob = next_log_probs[continuation_id].item()
    assert abs(continuation_log_prob - EXPECTED_LOG_PROB) < 5e-4, (
        f"Unexpected reference log-probability: {continuation_log_prob:.6f}"
    )

    top_log_probs, top_ids = torch.topk(next_log_probs, k=5)
    print(f"Prompt: {PROMPT!r}")
    print(f"Maximum absolute logit difference: {max_difference:.3e}")
    print(f"log P({CONTINUATION!r} | prompt): {continuation_log_prob:.6f}")
    print("Top five next tokens:")
    for token_id, log_prob in zip(top_ids.tolist(), top_log_probs.tolist()):
        print(f"  {tokenizer.decode([token_id])!r:16} log-probability={log_prob:.6f}")
    print("PASS: NanoLM reproduces the official GPT-2 forward pass.")


if __name__ == "__main__":
    main()
