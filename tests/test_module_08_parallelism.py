import pytest
from nanolm.parallelism_calc import (
    ModelSpecs, ParallelismConfig, calculate_memory_and_scaling,
    compute_param_parts, compute_pipeline_bubble_fraction, ring_allreduce_sent_bytes,
)


def result(**kwargs):
    return calculate_memory_and_scaling(ModelSpecs('toy', 4, 4, 64, 32, n_experts=4),
                                        ParallelismConfig(**kwargs))


def test_data_parallel_replication_does_not_save_state_memory():
    assert result(dp_size=4)['persistent_state_bytes'] == result()['persistent_state_bytes']
    assert result(dp_size=4)['global_tokens_per_step'] == 4 * result()['global_tokens_per_step']


def test_fsdp_shards_each_persistent_state_without_adding_devices():
    replicated = result(dp_size=4)
    sharded = result(dp_size=4, fsdp_size=4)
    assert sharded['total_gpus'] == replicated['total_gpus'] == 4
    for key in ('weights_bytes', 'gradients_bytes', 'optimizer_bytes'):
        assert sharded[key] == replicated[key] / 4


def test_ep_only_shards_experts_and_reuses_data_parallel_devices():
    specs = ModelSpecs('toy', 4, 4, 64, 32, n_experts=4)
    shared, experts = compute_param_parts(specs)
    ep = result(dp_size=4, ep_size=4)
    assert ep['total_gpus'] == 4
    assert ep['local_parameters_before_fsdp'] == shared + experts / 4
    assert ep['local_parameters_before_fsdp'] > (shared + experts) / 4


def test_bubble_threshold_and_communication_convention():
    assert compute_pipeline_bubble_fraction(4, 16) == pytest.approx(3/19)
    assert compute_pipeline_bubble_fraction(4, 27) == pytest.approx(.1)
    assert compute_pipeline_bubble_fraction(4, 28) < .1
    assert compute_pipeline_bubble_fraction(1, 1) == 0
    assert ring_allreduce_sent_bytes(1024, 4) == 1536


@pytest.mark.parametrize('config', [dict(dp_size=3, fsdp_size=2), dict(ep_size=2),
    dict(dp_size=4, fsdp_size=2, ep_size=2), dict(microbatches=0), dict(tp_size=3)])
def test_reject_undefined_or_invalid_layouts(config):
    with pytest.raises(ValueError):
        result(**config)
