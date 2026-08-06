# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
from dataclasses import fields
from megatron.core.models.gpt.gpt_layer_specs import (get_gpt_decoder_block_spec, get_gpt_mtp_block_spec)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from mcore_bridge import ModelConfig
from mcore_bridge import get_mcore_model as _get_mcore_model
from mcore_bridge import hf_to_mcore_config
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger

logger = get_logger()


def _disable_transformer_engine_for_alignment() -> bool:
    """Return whether alignment-mode model construction should avoid TE specs."""

    return os.environ.get('SWIFT_MEGATRON_NO_TE', '').lower() in {'1', 'true', 'yes', 'on'}


def _patch_no_te_duplicate_expert_chunks() -> None:
    """Keep no-TE EP expert outputs invariant to exact repeated chunks."""

    try:
        from megatron.core.transformer.moe.experts import SequentialMLP
    except Exception as e:
        logger.warning(f'Unable to patch no-TE duplicate expert chunks: {e!r}')
        return
    if getattr(SequentialMLP, '_swift_no_te_duplicate_chunk_patch', False):
        return

    def patched_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), '`moe_apply_probs_on_input` only works with `moe_router_topk`=1.'
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        if self.num_local_experts == 1:
            if self.config.fp8:
                hidden, probs = self._pad_tensor_for_fp8(permuted_local_hidden_states, permuted_probs)
                output, output_bias = self.local_experts[0](hidden, probs)
                output = output[:permuted_local_hidden_states.shape[0]]
            else:
                output, output_bias = self.local_experts[0](permuted_local_hidden_states, permuted_probs)
            return output, output_bias

        tokens_per_expert = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
        probs_list = torch.split(permuted_probs, tokens_per_expert)
        output_local_list = []

        for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
            repeat_count = 1
            dedup_tokens, dedup_probs = tokens, probs
            if not self.config.fp8 and tokens.shape[0] > 0:
                token_count = tokens.shape[0]
                max_repeat = min(token_count, int(getattr(self.config, 'expert_model_parallel_size', 1) or 1))
                for candidate_repeat in range(max_repeat, 1, -1):
                    if token_count % candidate_repeat != 0:
                        continue
                    chunk_size = token_count // candidate_repeat
                    base_tokens = tokens[:chunk_size]
                    base_probs = probs[:chunk_size]
                    chunks_equal = True
                    for chunk_idx in range(1, candidate_repeat):
                        start = chunk_idx * chunk_size
                        end = start + chunk_size
                        if not (torch.equal(base_tokens, tokens[start:end])
                                and torch.equal(base_probs, probs[start:end])):
                            chunks_equal = False
                            break
                    if chunks_equal:
                        repeat_count = candidate_repeat
                        dedup_tokens, dedup_probs = base_tokens, base_probs
                        break
            if repeat_count > 1:
                output, output_bias = expert(dedup_tokens, dedup_probs)
                output = torch.cat([output] * repeat_count, dim=0)
            elif self.config.fp8:
                hidden, probs = self._pad_tensor_for_fp8(tokens, probs)
                output, output_bias = expert(hidden, probs)
                output = output[:tokens.shape[0]]
            else:
                output, output_bias = expert(tokens, probs)
            assert output_bias is None, f'output_bias is not supported for {type(self).__name__}'
            output_local_list.append(output)

        return torch.cat(output_local_list, dim=0), None

    SequentialMLP.forward = patched_forward
    SequentialMLP._swift_no_te_duplicate_chunk_patch = True
    logger.info('Patched Megatron-Core SequentialMLP for no-TE duplicate expert chunks.')


class _NoTEMLADotProductAttention(DotProductAttention):
    """Compatibility wrapper for MLA local/no-TE attention spec."""

    def __init__(self, *args, k_channels=None, v_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        if k_channels is not None:
            self.hidden_size_per_attention_head = k_channels
        if v_channels is not None:
            self.hidden_size_per_partition = self.num_attention_heads_per_partition * v_channels


def _patch_no_te_mla_spec(transformer_layer_spec):
    layer_specs = getattr(transformer_layer_spec, 'layer_specs', None) or [transformer_layer_spec]
    for layer_spec in layer_specs:
        self_attn = getattr(getattr(layer_spec, 'submodules', None), 'self_attention', None)
        self_attn_submodules = getattr(self_attn, 'submodules', None)
        if self_attn_submodules is not None and hasattr(self_attn_submodules, 'core_attention'):
            self_attn_submodules.core_attention = _NoTEMLADotProductAttention
    return transformer_layer_spec


def _patch_no_te_mcore_bridge_specs() -> None:
    """Patch mcore_bridge to build local/no-TE specs under the alignment switch."""

    try:
        from mcore_bridge.model import register as mcore_register
    except Exception as e:
        logger.warning(f'Unable to patch mcore_bridge no-TE specs: {e!r}')
        return
    ModelLoader = mcore_register.ModelLoader
    if getattr(ModelLoader, '_swift_no_te_spec_patch', False):
        return

    def get_transformer_layer_spec(self, vp_stage=None):
        with self._patch_experimental_attention_variant():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                self.config,
                use_transformer_engine=False,
                normalization=self.config.normalization,
                qk_l2_norm=self.config.qk_l2_norm,
                vp_stage=vp_stage)
            self._deepcopy_layer_spec(transformer_layer_spec)
        if self.config.experimental_attention_variant == 'dsa':
            for layer_spec in transformer_layer_spec.layer_specs:
                self._replace_spec_dsa(layer_spec)
        return _patch_no_te_mla_spec(transformer_layer_spec)

    def get_mtp_block_spec(self, transformer_layer_spec, vp_stage=None):
        mtp_block_spec = get_gpt_mtp_block_spec(
            self.config, transformer_layer_spec, use_transformer_engine=False, vp_stage=vp_stage)
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.module = mcore_register.MultiTokenPredictionLayer
        return mtp_block_spec

    ModelLoader.get_transformer_layer_spec = get_transformer_layer_spec
    ModelLoader.get_mtp_block_spec = get_mtp_block_spec
    ModelLoader._swift_no_te_spec_patch = True
    logger.info('Patched mcore_bridge ModelLoader for no-TE alignment specs.')


def _check_attention_backend(args, config):
    """Validate attention backend compatibility with configuration."""
    attention_backend = config.attention_backend.name
    if attention_backend == 'flash' and config.softmax_type == 'learnable':
        raise ValueError(f'Attention backend "{attention_backend}" does not support learnable softmax_type.')


def _check_padding_free(args, config):
    """Validate and adjust padding_free setting based on configuration constraints."""
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    message = None

    if config.experimental_attention_variant == 'dsa':
        message = 'DSA is not supported in padding-free mode'
    elif attention_backend == 'unfused':
        message = f'Attention backend "{attention_backend}" is not supported in padding-free mode'

    if message:
        logger.warning(f'{message}. Setting args.padding_free to False.')
        args.padding_free = False


def get_mcore_model_config(args, hf_config):
    kwargs = hf_to_mcore_config(hf_config)
    kwargs['mcore_model_type'] = args.megatron_model_meta.model_type
    kwargs['hf_config'] = hf_config
    for f in fields(ModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['params_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp4_param'] = args.fp4_param_gather
    kwargs['fp8_param'] = args.fp8_param_gather
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    num_moe_experts = kwargs.get('num_moe_experts', None)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1
    if _disable_transformer_engine_for_alignment():
        kwargs['persist_layer_norm'] = False
        kwargs['moe_grouped_gemm'] = False
        if hasattr(args, 'moe_grouped_gemm'):
            args.moe_grouped_gemm = False

    if args.router_replay_mode != 'disabled':
        kwargs['moe_enable_routing_replay'] = True
    if args.megatron_extra_kwargs:
        kwargs.update(args.megatron_extra_kwargs)
    config = ModelConfig(**kwargs)
    if is_torch_npu_available() and getattr(args, 'attention_backend', 'flash') != 'local':
        setattr(config, 'use_flash_attn', True)
    _check_attention_backend(args, config)
    _check_padding_free(args, config)
    return config


def get_mcore_model(args, hf_config):
    if _disable_transformer_engine_for_alignment():
        _patch_no_te_mcore_bridge_specs()
        _patch_no_te_duplicate_expert_chunks()
    config = get_mcore_model_config(args, hf_config)
    models = _get_mcore_model(config)

    return models
