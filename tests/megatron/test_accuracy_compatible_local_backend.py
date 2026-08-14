# Copyright (c) ModelScope Contributors. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from swift.megatron.init import _patch_mcore_bridge_disable_te
from swift.megatron.model.utils import _audit_accuracy_compatible_model


class _Root(torch.nn.Module):
    def __init__(self, child=None):
        super().__init__()
        if child is not None:
            self.child = child


class _FakeTransformerEngineModule(torch.nn.Module):
    pass


_FakeTransformerEngineModule.__module__ = 'megatron.core.extensions.transformer_engine'


def _args():
    return SimpleNamespace(use_accuracy_compatible=True)


def test_accuracy_model_audit_accepts_local_model_without_te():
    args = _args()
    _audit_accuracy_compatible_model(args, SimpleNamespace(transformer_impl='local'), [_Root(torch.nn.Linear(2, 2))])

    assert args.repro_transformer_impl == 'local'
    assert args.repro_transformer_engine_module_count == 0
    assert args.repro_transformer_engine_module_classes == []


def test_accuracy_model_audit_rejects_non_local_config():
    with pytest.raises(RuntimeError, match='requires transformer_impl="local"'):
        _audit_accuracy_compatible_model(_args(), SimpleNamespace(transformer_impl='transformer_engine'), [_Root()])


def test_accuracy_model_audit_rejects_te_module():
    with pytest.raises(RuntimeError, match='forbids TransformerEngine modules'):
        _audit_accuracy_compatible_model(
            _args(), SimpleNamespace(transformer_impl='local'), [_Root(_FakeTransformerEngineModule())]
        )


def test_qwen3next_gdn_bridge_maps_standalone_input_norm_for_both_layer_types():
    _patch_mcore_bridge_disable_te()
    from mcore_bridge.model.gpts.qwen3_next_gdn import Qwen3NextGDNBridgeMixin

    calls = []
    bridge = SimpleNamespace(
        config=SimpleNamespace(linear_attention_freq=[1, 0]),
        hf_input_layernorm_key='input_layernorm.weight',
        _set_linear_attn_state=lambda module, state, prefix, index, to_mcore: {'linear': prefix},
        _set_attn_state=lambda module, state, prefix, index, to_mcore: {'full': prefix},
        _set_state_dict=lambda module, key, state, hf_key, to_mcore: calls.append((key, hf_key)),
    )
    layer = SimpleNamespace(self_attention=object())
    patched = Qwen3NextGDNBridgeMixin._set_layer_attn

    linear_state = patched(bridge, layer, {}, 0, True)
    full_state = patched(bridge, layer, {}, 1, True)

    assert linear_state == {'linear': 'linear_attn.'}
    assert full_state == {'full': 'self_attn.'}
    assert calls == [
        ('input_layernorm.weight', 'input_layernorm.weight'),
        ('input_layernorm.weight', 'input_layernorm.weight'),
    ]
