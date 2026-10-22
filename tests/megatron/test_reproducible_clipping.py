"""Explicit clipping survives accuracy-mode initialization."""

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from swift.megatron.pipelines.train import sft as sft_module
from swift.megatron.trainers import base as trainer_module
from swift.pipelines.base import SwiftPipeline


@pytest.mark.parametrize('accuracy', [False, True])
@pytest.mark.parametrize('clip', [0.0, 0.25, 1.0])
def test_constructor_preserves_explicit_clipping(accuracy, clip):
    args = SimpleNamespace(
        clip_grad=clip,
        use_accuracy_compatible=accuracy,
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
    assert instance.args.clip_grad == clip


def test_unsupported_megatron_rejects_requested_norm(monkeypatch):
    from megatron.core.optimizer import clip_grads
    trainer = SimpleNamespace(args=SimpleNamespace(use_accuracy_compatible=True, clip_grad=1.0))
    monkeypatch.delattr(clip_grads, 'get_reproducible_grad_norm_bins', raising=False)
    with pytest.raises(ValueError, match='Megatron-Core reproducible norm support'):
        trainer_module.BaseMegatronTrainer.get_optimizer_and_scheduler(trainer)
