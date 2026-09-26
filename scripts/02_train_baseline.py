"""Module 2 Part B: disjoint documents, fixed token budget, recoverable baseline."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import torch
from nanolm.baseline import BaselineConfig, run_baseline


def read_docs(path):
    return [json.loads(line)['text'] for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-jsonl')
    parser.add_argument('--val-jsonl')
    parser.add_argument('--output', default='runs/baseline')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--threads', type=int, default=1)
    for field in fields(BaselineConfig):
        parser.add_argument('--' + field.name.replace('_', '-'), type=field.type, default=field.default)
    args = parser.parse_args()
    if bool(args.train_jsonl) != bool(args.val_jsonl):
        parser.error('Supply both --train-jsonl and --val-jsonl, or neither for the smoke fixture')
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.train_jsonl:
        train, val = read_docs(args.train_jsonl), read_docs(args.val_jsonl)
    else:
        print('Synthetic smoke fixture: validates the workflow, not useful language quality.')
        train = [f'The {item} is in the room. We read and write a short sentence about it. '
                 for item in ('book', 'table', 'chair', 'lamp', 'pencil', 'notebook')]
        val = [f'The {item} is near the window. We describe it in another sentence. '
               for item in ('plant', 'clock', 'painting')]
    cfg = BaselineConfig(**{f.name: getattr(args, f.name) for f in fields(BaselineConfig)})
    result = run_baseline(train, val, args.output, cfg, resume=args.resume, stop_after=args.stop_after)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
