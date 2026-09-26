"""Small, inspectable held-out baseline with reproducible CPU resume.

Fixed UTF-8 byte vocabulary (256 bytes + EOT) avoids network downloads and
validation-dependent vocabulary fitting. This is a baseline protocol, not a
claim that the default synthetic fixture produces useful language quality.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from .data import BinaryShardedDataset, pack_documents
from .model import GPTConfig, MiniGPT
from .optim import configure_optimizers, get_lr_cosine_schedule
from .tokenizer import BPETokenizer
from .train import train_step, optimizer_step, save_checkpoint, load_checkpoint


@dataclass(frozen=True)
class BaselineConfig:
    steps: int = 60
    batch_size: int = 4
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    learning_rate: float = 0.002
    seed: int = 42
    eval_every: int = 10
    eval_batches: int = 4
    device: str = "cpu"


def document_manifest(train_docs, val_docs):
    """Reject empty documents and normalized exact duplicates, including leakage."""
    seen = set()
    manifest = {}
    for split, docs in (("train", train_docs), ("validation", val_docs)):
        if not docs:
            raise ValueError(f"{split} must contain documents")
        hashes = []
        for text in docs:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Documents must be nonempty strings")
            digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            if digest in seen:
                raise ValueError("Duplicate document within or across splits")
            seen.add(digest)
            # Exact bytes also matter for reproducibility; normalization is only a leak check.
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
        manifest[split] = hashes
    return manifest


@torch.no_grad()
def held_out_loss(model, dataset, cfg):
    """Fixed contiguous validation windows; preserve the caller's train/eval mode."""
    previous = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    try:
        for index, (x, y) in enumerate(dataset.iterate_batches(1, cfg.block_size, torch.device(cfg.device))):
            if index >= cfg.eval_batches:
                break
            _, loss, _ = model(x, targets=y)
            loss_sum += loss.item() * y.numel()
            token_count += y.numel()
    finally:
        model.train(previous)
    if not token_count:
        raise ValueError("Validation shard must contain at least block_size + 1 tokens")
    return loss_sum / token_count


def run_baseline(train_docs, val_docs, output_dir, cfg=BaselineConfig(), *, resume=None, stop_after=None):
    """Run to cfg.steps or stop_after; cfg.steps stays fixed across resume.

    CPU continuation is tested exactly. Other devices/releases need their own
    deterministic-kernel checks; the function makes no cross-device guarantee.
    """
    if min(cfg.steps, cfg.batch_size, cfg.block_size, cfg.eval_every, cfg.eval_batches,
           cfg.n_layer, cfg.n_head, cfg.n_embd) < 1 or cfg.learning_rate <= 0:
        raise ValueError("Training dimensions, cadence, and learning rate must be positive")
    end = cfg.steps if stop_after is None else stop_after
    if not 0 < end <= cfg.steps:
        raise ValueError("stop_after must be between 1 and the planned steps")
    manifest = document_manifest(train_docs, val_docs)
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        extra = saved.get('extra_state', {})
        if extra.get('run_config') != asdict(cfg) or extra.get('manifest') != manifest:
            raise ValueError("Resume configuration or document manifest differs from checkpoint")
        if saved['step'] >= end:
            raise ValueError("Requested stopping step must be after checkpoint step")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / 'checkpoint.pt'
    if checkpoint_path.exists() and resume is None:
        raise ValueError("Output already contains a checkpoint; use --resume or a new directory")
    tokenizer = BPETokenizer()
    tokenizer.train('', vocab_size=257)  # byte IDs 0..255, EOT 256, no learned merges
    for split, docs in (("train", train_docs), ("validation", val_docs)):
        pack_documents(docs, tokenizer, str(output / f'{split}.bin'), tokenizer.eot_token_id)
    train = BinaryShardedDataset(str(output / 'train.bin'))
    validation = BinaryShardedDataset(str(output / 'validation.bin'))
    if min(train.num_tokens, validation.num_tokens) <= cfg.block_size:
        raise ValueError("Each split needs more tokens than block_size")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model_cfg = GPTConfig(vocab_size=257, block_size=cfg.block_size, n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_embd=cfg.n_embd)
    model = MiniGPT(model_cfg).to(cfg.device)
    optimizer = configure_optimizers(model, learning_rate=cfg.learning_rate, device_type=cfg.device)
    records = []
    start = 0
    if resume is not None:
        saved = load_checkpoint(str(resume), model, optimizer, restore_rng=True)
        start, records = saved['step'], saved['extra_state']['records']
    else:
        records.append({'step': 0, 'tokens': 0, 'validation_loss': held_out_loss(model, validation, cfg)})

    last_val = records[-1]['validation_loss']
    for step in range(start, end):
        lr = get_lr_cosine_schedule(step, min(5, cfg.steps - 1), cfg.steps, cfg.learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = lr
        x, y = train.get_batch(cfg.batch_size, cfg.block_size, torch.device(cfg.device))
        loss = train_step(model, optimizer, x, y)
        norm = optimizer_step(model, optimizer)
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            last_val = held_out_loss(model, validation, cfg)
            records.append({'step': step + 1, 'tokens': (step + 1)*cfg.batch_size*cfg.block_size,
                            'train_loss': loss, 'validation_loss': last_val, 'grad_norm': norm, 'lr': lr})
    save_checkpoint(str(checkpoint_path), model, optimizer, model_cfg, end, last_val,
                    extra_state={'run_config': asdict(cfg), 'manifest': manifest, 'records': records})
    report = {'config': asdict(cfg), 'tokenizer': 'utf8-bytes-256-plus-eot', 'manifest': manifest,
              'completed_steps': end, 'completed_tokens': end*cfg.batch_size*cfg.block_size,
              'records': records, 'torch_version': torch.__version__,
              'validation_protocol': f'first {cfg.eval_batches} full windows of {cfg.block_size} tokens',
              'checkpoint': str(checkpoint_path)}
    try:
        repo = Path(__file__).resolve().parents[1]
        report['git_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        report['git_dirty'] = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        report['git_commit'] = None
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
