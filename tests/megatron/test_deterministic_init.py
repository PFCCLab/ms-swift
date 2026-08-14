from types import SimpleNamespace

import torch

from swift.megatron.utils import megatron_lm_utils


def test_initialize_megatron_enables_global_determinism(monkeypatch):
    monkeypatch.setattr(megatron_lm_utils, '_initialize_mpu', lambda args: None)
    monkeypatch.setattr(megatron_lm_utils, 'set_random_seed', lambda *args, **kwargs: None)
    args = SimpleNamespace(
        deterministic_mode=True,
        seed=1234,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        model_info=SimpleNamespace(is_moe_model=False),
    )

    torch.use_deterministic_algorithms(False)
    try:
        megatron_lm_utils.initialize_megatron(args)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)
