"""Module 6 (session 21): full versus sharded checkpoints, reloaded in ONE process.

First produce both formats with two FSDP2 ranks:

    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py --save runs/fsdp2-ckpt

Then inspect and reload them without any process group:

    uv run python scripts/06_checkpoint_formats.py runs/fsdp2-ckpt

The sharded checkpoint was written by two ranks, one file each. Loading it into
a single unsharded model is a *resharding* load: the distributed-checkpoint
metadata records which slice of every tensor each file holds.
"""

import argparse
import warnings
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from nanolm.model import GPTConfig, MiniGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="directory written by 06_fsdp2.py --save")
    args = parser.parse_args()
    full_path, sharded_dir = args.checkpoint / "full.pt", args.checkpoint / "sharded"

    print("files on disk:")
    for path in [full_path, *sorted(sharded_dir.iterdir())]:
        print(f"  {str(path):45s} {path.stat().st_size:>9,d} B")

    metadata = dcp.FileSystemReader(str(sharded_dir)).read_metadata()
    model_keys = [k for k in metadata.state_dict_metadata if k.startswith("model.")]
    optim_keys = [k for k in metadata.state_dict_metadata if k.startswith("optimizer.")]
    print(f"\nsharded metadata: {len(model_keys)} model tensors, {len(optim_keys)} optimizer entries")
    example = "model.transformer.h.0.mlp.c_fc.weight"
    if example in metadata.state_dict_metadata:
        chunks = metadata.state_dict_metadata[example].chunks
        print(f"  {example}: global {tuple(metadata.state_dict_metadata[example].size)}, "
              + ", ".join(f"slice at {tuple(c.offsets)} size {tuple(c.sizes)}" for c in chunks))

    saved = torch.load(full_path, map_location="cpu")
    cfg = GPTConfig(**saved["config"])
    from_full = MiniGPT(cfg)
    from_full.load_state_dict(saved["model"])

    from_sharded = MiniGPT(cfg)
    state = {"model": from_sharded.state_dict()}
    with warnings.catch_warnings():  # loading without a process group is the point here
        warnings.filterwarnings("ignore", message=".*single process.*")
        dcp.load(state, checkpoint_id=str(sharded_dir))  # one reader, all files
    from_sharded.load_state_dict(state["model"])

    difference = max((a - b).abs().max().item()
                     for a, b in zip(from_full.state_dict().values(), from_sharded.state_dict().values()))
    print(f"\nmax |full - resharded| over all tensors = {difference:.1e}")
    x = torch.arange(8).view(1, 8) % cfg.vocab_size
    with torch.no_grad():
        same = torch.equal(from_full(x)[0], from_sharded(x)[0])
    print(f"identical logits from both reloads: {same}")
    if difference != 0.0 or not same:
        raise SystemExit("FAIL: the two checkpoint formats do not describe the same model")
    print("PASS: a 2-rank sharded checkpoint reloads into 1 process and equals the gathered full checkpoint.")
    print("Not covered: optimizer reload, data-loader position, RNG state, and a resumed trajectory.")


if __name__ == "__main__":
    main()
