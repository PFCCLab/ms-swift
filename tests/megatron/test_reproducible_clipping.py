"""Explicit clipping survives accuracy-mode initialization."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swift.megatron.pipelines.train import sft as sft_module
from swift.megatron.trainers import base as trainer_module
from swift.pipelines.base import SwiftPipeline


@pytest.mark.parametrize('accuracy,enabled,clip,expected', [
    (True, True, 1.0, 1.0),
    (True, True, 0.25, 0.25),
    (True, True, 0.0, 0.0),
    (True, False, 1.0, 0.0),
    (False, False, 1.0, 1.0),
])
def test_constructor_preserves_explicit_clipping(accuracy, enabled, clip, expected):
    args = SimpleNamespace(
        clip_grad=clip,
        reproducible_grad_norm=enabled,
        template_meta=SimpleNamespace(template_cls=None),
        model_meta=SimpleNamespace(is_multimodal=False),
        mcore_model='existing-model',
        output_dir='unused',
        get_model_processor=lambda **kwargs: (None, None),
        save_args=lambda output_dir: None)

    def pipeline_init(instance, values):
        instance.args = values

    def prepare_template(instance):
        instance.template = SimpleNamespace()

    with patch.object(SwiftPipeline, '__init__', pipeline_init), \
            patch.object(sft_module, 'repatch', None), \
            patch.object(sft_module.MegatronSft, '_prepare_template', prepare_template), \
            patch('megatron.core.transformer.module._use_accuracy_compatible', return_value=accuracy):
        instance = sft_module.MegatronSft(args)
    assert instance.args.clip_grad == expected


def test_unsupported_megatron_rejects_requested_norm():
    trainer = SimpleNamespace(args=SimpleNamespace(reproducible_grad_norm=True))
    with patch.object(trainer_module, 'mcore_016', False), \
            patch.object(trainer_module, 'OptimizerConfig', type('OldConfig', (), {})), \
            pytest.raises(ValueError, match='Megatron-Core version'):
        trainer_module.BaseMegatronTrainer.get_optimizer_and_scheduler(trainer)
