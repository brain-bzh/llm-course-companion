import pytest
from nanolm.model import GPTConfig
from nanolm.profile_utils import compute_flops_per_token, estimate_mfu


def test_flops_include_output_head_and_attention_length():
    a = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)
    b = GPTConfig(vocab_size=200, block_size=32, n_layer=2, n_head=2, n_embd=32)
    assert compute_flops_per_token(b)-compute_flops_per_token(a) == 6*100*32 + 12*2*32*16


def test_mfu_units_and_invalid_peak():
    assert estimate_mfu(1000, 1_000_000_000, 10) == 10
    with pytest.raises(ValueError):
        estimate_mfu(1000, 1000, 0)


def test_attention_check_detects_missing_sdpa_path(monkeypatch):
    """A model that ignores use_sdpa must not produce a false comparison."""
    import importlib.util
    from pathlib import Path
    import torch

    path = Path(__file__).resolve().parents[1] / "scripts" / "04_check_execution.py"
    spec = importlib.util.spec_from_file_location("check_execution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    module.check_attention(torch.device("cpu"))
    model_type = module.MiniGPT

    def ignore_sdpa(config):
        config.use_sdpa = False
        return model_type(config)

    monkeypatch.setattr(module, "MiniGPT", ignore_sdpa)
    with pytest.raises(AssertionError, match="check that your model honors use_sdpa"):
        module.check_attention(torch.device("cpu"))


def test_static_memory_is_sixteen_bytes_per_fp32_parameter_with_adamw():
    import torch
    from nanolm.model import MiniGPT
    from nanolm.profile_utils import memory_breakdown

    model = MiniGPT(GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=16))
    n = sum(p.numel() for p in model.parameters())  # tied weight counted once
    mb = memory_breakdown(model, torch.optim.AdamW(model.parameters()))
    assert mb["parameters_mb"] * 2**20 == 4 * n
    assert mb["optimizer_states_mb"] * 2**20 == 8 * n
    assert mb["total_static_mb"] * 2**20 == 16 * n
