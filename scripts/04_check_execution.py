"""Check manual/SDPA and optional compiled NanoLM outputs and gradients.

This is a fixed-batch FP32 check, not a proof for all inputs or precisions.
"""

import argparse
import copy
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanolm.model import GPTConfig, MiniGPT


def compare_models(reference, candidate, x, y):
    """Compare forward values and parameter gradients from identical models."""
    logits_a, loss_a, _ = reference(x, targets=y)
    logits_b, loss_b, _ = candidate(x, targets=y)
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-3, atol=1e-4)
    loss_a.backward()
    loss_b.backward()
    params_a = dict(reference.named_parameters())
    params_b = dict(candidate.named_parameters())
    if params_a.keys() != params_b.keys():
        raise AssertionError("Parameter names differ between compared models")
    for name in params_a:
        a, b = params_a[name].grad, params_b[name].grad
        if a is None or b is None:
            if a is not None or b is not None:
                raise AssertionError(f"Gradient presence differs for {name}")
        else:
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4, msg=lambda msg: f"{name}: {msg}")


def check_attention(device):
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    manual = MiniGPT(cfg).to(device)
    sdpa_cfg = copy.deepcopy(cfg)
    sdpa_cfg.use_sdpa = True
    sdpa = MiniGPT(sdpa_cfg).to(device)
    sdpa.load_state_dict(manual.state_dict())
    tokens = torch.randint(0, cfg.vocab_size, (2, 17), device=device)
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as called:
        compare_models(manual, sdpa, x, y)
        if called.call_count != cfg.n_layer:
            raise AssertionError(
                f"Expected SDPA in each of {cfg.n_layer} blocks, observed {called.call_count} calls; "
                "check that your model honors use_sdpa."
            )
    return cfg, sdpa, x, y


def check_compile(sdpa, x, y):
    eager = copy.deepcopy(sdpa)
    eager.zero_grad(set_to_none=True)
    compiled_raw = copy.deepcopy(eager)
    compiled = torch.compile(compiled_raw)
    # Compiled wrapper prefixes parameter names, so compare gradients on its raw model.
    logits_a, loss_a, _ = eager(x, targets=y)
    logits_b, loss_b, _ = compiled(x, targets=y)
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-3, atol=1e-4)
    loss_a.backward()
    loss_b.backward()
    for (name_a, a), (name_b, b) in zip(eager.named_parameters(), compiled_raw.named_parameters(), strict=True):
        if name_a != name_b or (a.grad is None) != (b.grad is None):
            raise AssertionError(f"Parameter or gradient mismatch: {name_a}, {name_b}")
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-3, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compile", action="store_true", help="Also compare eager and torch.compile SDPA")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    _, sdpa, x, y = check_attention(torch.device(args.device))
    print("Manual and SDPA logits, loss and gradients agree; SDPA was called.")
    if args.compile:
        check_compile(sdpa, x, y)
        print("Eager and compiled SDPA logits, loss and gradients agree.")


if __name__ == "__main__":
    main()
