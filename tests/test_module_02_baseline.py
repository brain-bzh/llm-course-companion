from dataclasses import replace
import pytest
import torch
from nanolm.baseline import BaselineConfig, document_manifest, run_baseline

TRAIN = ['Training text with a distinct document identity. ' * 3,
         'Another document teaches the model about letters. ' * 3]
VAL = ['Held-out words are measured separately from training. ' * 3]
CFG = BaselineConfig(steps=4, batch_size=2, block_size=8, n_layer=1, n_head=2,
                     n_embd=16, eval_every=2, eval_batches=2)


def test_reject_leakage_and_empty_data():
    with pytest.raises(ValueError, match='Duplicate'):
        document_manifest(['same text'], ['same   text'])
    with pytest.raises(ValueError):
        document_manifest([], VAL)


def test_interrupted_training_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    full = run_baseline(TRAIN, VAL, tmp_path/'full', CFG)
    run_baseline(TRAIN, VAL, tmp_path/'resumed', CFG, stop_after=2)
    resumed = run_baseline(TRAIN, VAL, tmp_path/'resumed', CFG,
                           resume=tmp_path/'resumed/checkpoint.pt')
    assert full['records'] == resumed['records']
    assert resumed['completed_tokens'] == 4*2*8
    a = torch.load(tmp_path/'full/checkpoint.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for param in a['optimizer']['state']:
        for key in a['optimizer']['state'][param]:
            torch.testing.assert_close(a['optimizer']['state'][param][key],
                                       b['optimizer']['state'][param][key], rtol=0, atol=0)


def test_resume_rejects_changed_data_or_schedule(tmp_path):
    run_baseline(TRAIN, VAL, tmp_path, CFG, stop_after=2)
    for docs, cfg in ((VAL + ['new document'], CFG), (VAL, replace(CFG, steps=5))):
        with pytest.raises(ValueError, match='differs'):
            run_baseline(TRAIN, docs, tmp_path, cfg, resume=tmp_path/'checkpoint.pt')
