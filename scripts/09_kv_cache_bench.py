"""Module 9: fixed-workload KV latency; no guaranteed speedup."""
import argparse
import json
import torch
from nanolm.model import MiniGPT, GPTConfig
from nanolm.kv_cache import benchmark_generation_speed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--prompt-tokens', type=int, default=16)
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=256, block_size=args.prompt_tokens+args.new_tokens,
                    n_layer=2, n_head=2, n_embd=64)
    model = MiniGPT(cfg).to(args.device)
    prompt = torch.randint(0, cfg.vocab_size, (args.batch_size, args.prompt_tokens), device=args.device)
    print(json.dumps(benchmark_generation_speed(model, prompt, args.new_tokens, args.repeats), indent=2))


if __name__ == '__main__':
    main()
