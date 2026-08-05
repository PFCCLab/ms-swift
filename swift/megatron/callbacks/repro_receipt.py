# Copyright (c) ModelScope Contributors. All rights reserved.
"""Machine-readable receipts for model-reproduction runs.

This callback is intentionally opt-in. It records raw (unrounded) per-step loss,
the actual runtime fingerprint, and a regular-file safetensors checkpoint copy in
stable locations under ``args.output_dir``.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from swift.utils import is_last_rank

from .base import MegatronCallback


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _torch_input_array(value):
    if not torch.is_tensor(value):
        return None
    original_shape = list(value.shape)
    original_stride = list(value.stride())
    logical_dtype = str(value.dtype).replace('torch.', '')
    value = value.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        array = value.view(torch.uint16).numpy()
        storage_dtype = 'uint16'
    else:
        array = value.numpy()
        storage_dtype = str(array.dtype)
    return array, original_shape, original_stride, logical_dtype, storage_dtype


def _write_model_input_receipt(output_dir, framework, rank, inputs, labels, step, phase):
    receipt_dir = output_dir / 'model_inputs'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    tensors = {}
    for name in sorted(inputs):
        if name == 'labels':
            continue
        converted = _torch_input_array(inputs[name])
        if converted is None:
            continue
        array, shape, stride, logical_dtype, storage_dtype = converted
        array_name = f't{len(arrays)}'
        arrays[array_name] = array
        tensors[name] = {
            'array': array_name,
            'shape': shape,
            'stride': stride,
            'dtype': logical_dtype,
            'storage_dtype': storage_dtype,
            'sha256': __import__('hashlib').sha256(array.tobytes(order='C')).hexdigest(),
        }
    converted = _torch_input_array(labels)
    if converted is not None:
        array, shape, stride, logical_dtype, storage_dtype = converted
        array_name = f't{len(arrays)}'
        arrays[array_name] = array
        tensors['labels'] = {
            'array': array_name,
            'shape': shape,
            'stride': stride,
            'dtype': logical_dtype,
            'storage_dtype': storage_dtype,
            'sha256': __import__('hashlib').sha256(array.tobytes(order='C')).hexdigest(),
        }
    npz_path = receipt_dir / f'model_inputs_rank{rank}.npz'
    np.savez(npz_path, **arrays)
    manifest = {
        'schema': 'model-facing-input-receipt/v1',
        'framework': framework,
        'rank': rank,
        'step': int(step),
        'phase': phase,
        'tensor_count': len(tensors),
        'tensors': tensors,
        'npz': npz_path.name,
    }
    _atomic_json(receipt_dir / f'model_inputs_rank{rank}.json', manifest)


def _first_torch_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_torch_tensor(item)
            if tensor is not None:
                return tensor
    elif isinstance(value, dict):
        if "hidden_states" in value:
            tensor = _first_torch_tensor(value["hidden_states"])
            if tensor is not None:
                return tensor
        for key in sorted(value):
            tensor = _first_torch_tensor(value[key])
            if tensor is not None:
                return tensor
    return None


def _first_torch_call_tensor(inputs, kwargs):
    tensor = _first_torch_tensor(inputs)
    if tensor is not None:
        return tensor
    return _first_torch_tensor(kwargs)


def _write_named_internal_receipt(output_dir, framework, rank, boundary, module_name, module_class, calls):
    receipt_dir = output_dir / 'internal_boundaries'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    call_receipts = []
    for call_index, call in enumerate(calls):
        tensors = {}
        for role, converted in call.items():
            array, shape, stride, logical_dtype, storage_dtype = converted
            array_name = f'c{call_index}_{role}'
            arrays[array_name] = array
            tensors[role] = {
                'array': array_name,
                'shape': shape,
                'stride': stride,
                'dtype': logical_dtype,
                'storage_dtype': storage_dtype,
                'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            }
        call_receipts.append({'call_index': call_index, 'tensors': tensors})
    npz_path = receipt_dir / f'{boundary}_rank{rank}.npz'
    np.savez(npz_path, **arrays)
    _atomic_json(
        receipt_dir / f'{boundary}_rank{rank}.json',
        {
            'schema': 'internal-component-receipt/v1',
            'framework': framework,
            'rank': rank,
            'step': 0,
            'boundary': boundary,
            'module_name': module_name,
            'module_class': module_class,
            'call_count': len(call_receipts),
            'calls': call_receipts,
            'npz': npz_path.name,
        },
    )


def _write_internal_boundary_receipt(output_dir, rank, boundary, module_name, module_class, calls):
    receipt_dir = output_dir / 'internal_boundaries'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    call_receipts = []
    for call_index, call in enumerate(calls):
        call_receipt = {'call_index': call_index}
        for role in ('input', 'output'):
            array, shape, stride, logical_dtype, storage_dtype = call[role]
            array_name = f'c{call_index}_{role}'
            arrays[array_name] = array
            call_receipt[role] = {
                'array': array_name,
                'shape': shape,
                'stride': stride,
                'dtype': logical_dtype,
                'storage_dtype': storage_dtype,
                'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            }
        call_receipts.append(call_receipt)
    npz_path = receipt_dir / f'{boundary}_rank{rank}.npz'
    np.savez(npz_path, **arrays)
    _atomic_json(
        receipt_dir / f'{boundary}_rank{rank}.json',
        {
            'schema': 'internal-boundary-receipt/v1',
            'framework': 'torch',
            'rank': rank,
            'step': 0,
            'boundary': boundary,
            'module_name': module_name,
            'module_class': module_class,
            'call_count': len(call_receipts),
            'calls': call_receipts,
            'npz': npz_path.name,
        },
    )


class ReproReceiptCallback(MegatronCallback):
    """Write the stable ``loss.json``, ``env.json`` and ``checkpoint/`` contract."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.is_write_rank = is_last_rank()
        self.output_dir = Path(self.args.output_dir)
        self.losses_by_iteration = {}
        self._model_inputs_captured = False
        self.capture_model_inputs = os.environ.get("REPRO_CAPTURE_MODEL_INPUTS", "1") == "1"
        self.internal_boundary = os.environ.get('REPRO_CAPTURE_INTERNAL_BOUNDARIES', '').strip()
        self._internal_boundary_calls = []
        self._internal_boundary_handle = None
        self._internal_boundary_restore = None

    def on_train_begin(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                shutil.rmtree(self.output_dir / 'model_inputs', ignore_errors=True)
            torch.distributed.barrier()
        else:
            shutil.rmtree(self.output_dir / 'model_inputs', ignore_errors=True)
        self._setup_internal_boundary_receipt()
        if not self.is_write_rank:
            return
        model_dir = Path(self.args.model_dir)
        profile_path = model_dir / 'repro_profile.json'
        if not profile_path.is_file():
            raise FileNotFoundError(f'repro receipt requires {profile_path}')
        profile = json.loads(profile_path.read_text(encoding='utf-8'))
        config_path = model_dir / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        nccl_version = torch.cuda.nccl.version() if torch.cuda.is_available() else None
        env = {
            'framework': 'torch',
            'framework_version': torch.__version__,
            'device': 'cuda',
            'device_name': torch.cuda.get_device_name(torch.cuda.current_device()),
            'dtype': 'bfloat16',
            'model_id': profile['source_model_id'],
            'revision': profile['source_revision'],
            'model_config_sha256': profile['source_config_sha256'],
            'profile_config_sha256': _sha256_file(config_path),
            'profile': profile['schema'],
            'profile_tensor_count': profile['tensor_count'],
            'profile_payload_bytes': profile['payload_bytes'],
            'weights_loaded': True,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
            'torch_compile_disable': os.environ.get('TORCH_COMPILE_DISABLE'),
            'torch_cuda': torch.version.cuda,
            'nccl_version': nccl_version,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'world_size': world_size,
            'tensor_model_parallel_size': self.args.tensor_model_parallel_size,
            'pipeline_model_parallel_size': self.args.pipeline_model_parallel_size,
            'context_parallel_size': self.args.context_parallel_size,
            'expert_model_parallel_size': self.args.expert_model_parallel_size,
            'expert_tensor_parallel_size': self.args.expert_tensor_parallel_size,
            'sequence_parallel': self.args.sequence_parallel,
            'use_distributed_optimizer': self.args.use_distributed_optimizer,
            'use_accuracy_compatible': self.args.use_accuracy_compatible,
            'transformer_impl': getattr(self.args, 'repro_transformer_impl', None),
            'transformer_engine_module_count': getattr(self.args, 'repro_transformer_engine_module_count', None),
            'transformer_engine_module_classes': getattr(self.args, 'repro_transformer_engine_module_classes', None),
            'train_iters': self.args.train_iters,
            'seed': self.args.seed,
            'optimizer': self.args.optimizer,
            'learning_rate': self.args.lr,
            'adam_beta1': self.args.adam_beta1,
            'adam_beta2': self.args.adam_beta2,
            'adam_eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'clip_grad': self.args.clip_grad,
        }
        _atomic_json(self.output_dir / 'env.json', env)

    def _write_losses(self):
        ordered = [self.losses_by_iteration[i] for i in sorted(self.losses_by_iteration)]
        _atomic_json(self.output_dir / 'loss.json', {'framework': 'torch', 'losses': ordered})

    def _setup_internal_boundary_receipt(self):
        if not self.internal_boundary:
            return
        target_classes_by_boundary = {
            'text_embedding_lookup': {'VocabParallelEmbedding'},
            'language_layer0': {'TransformerLayer'},
            'language_layer0_input_norm': {'TorchRMSNorm'},
            'language_layer0_gdn': {'GatedDeltaNet'},
            'language_layer0_gdn_in_proj': {'ColumnParallelLinear'},
            'language_layer0_gdn_recurrence': {'GatedDeltaNet'},
            'language_layer0_gdn_out_norm': {'TorchRMSNorm'},
            'language_layer0_gdn_out_proj': {'RowParallelLinear'},
            'vision_patch_embed': {'Qwen3_5VisionPatchEmbed', 'Qwen3_5MoeVisionPatchEmbed'},
            'vision_pos_lookup': {'Embedding'},
            'vision_block0': {'Qwen3_5VisionBlock', 'Qwen3_5MoeVisionBlock'},
            'vision_merger': {'Qwen3_5VisionPatchMerger', 'Qwen3_5MoeVisionPatchMerger'},
        }
        if self.internal_boundary not in target_classes_by_boundary:
            raise ValueError(f'unsupported internal reproduction boundary: {self.internal_boundary!r}')
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        receipt_dir = self.output_dir / 'internal_boundaries'
        if rank == 0:
            shutil.rmtree(receipt_dir, ignore_errors=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        matches = []
        target_classes = target_classes_by_boundary[self.internal_boundary]
        for model_index, model in enumerate(self.trainer.unwrapped_models):
            for name, module in model.named_modules():
                if type(module).__name__ not in target_classes:
                    continue
                if self.internal_boundary == 'language_layer0' and not name.endswith(
                    'language_model.decoder.layers.0'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_input_norm' and not name.endswith(
                    'language_model.decoder.layers.0.input_layernorm'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_gdn' and not name.endswith(
                    'language_model.decoder.layers.0.self_attention'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_gdn_in_proj' and not name.endswith(
                    'language_model.decoder.layers.0.self_attention.in_proj'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_gdn_recurrence' and not name.endswith(
                    'language_model.decoder.layers.0.self_attention'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_gdn_out_norm' and not name.endswith(
                    'language_model.decoder.layers.0.self_attention.out_norm'
                ):
                    continue
                if self.internal_boundary == 'language_layer0_gdn_out_proj' and not name.endswith(
                    'language_model.decoder.layers.0.self_attention.out_proj'
                ):
                    continue
                if self.internal_boundary == 'vision_pos_lookup' and not name.endswith('visual.pos_embed'):
                    continue
                if self.internal_boundary == 'vision_block0' and not name.endswith('visual.blocks.0'):
                    continue
                matches.append((f'model{model_index}.{name}', module))
        if len(matches) != 1:
            found = [f'{name}:{type(module).__module__}.{type(module).__qualname__}' for name, module in matches]
            raise RuntimeError(
                f'{self.internal_boundary} receipt requires exactly one runtime module, found {found}'
            )
        module_name, module = matches[0]
        module_class = f'{type(module).__module__}.{type(module).__qualname__}'

        if self.internal_boundary == 'language_layer0_gdn_recurrence':
            runtime_module = sys.modules.get(type(module).__module__)
            if runtime_module is None or not hasattr(runtime_module, 'torch_chunk_gated_delta_rule'):
                raise RuntimeError('GDN recurrence receipt could not locate the owning runtime function')
            original_rule = runtime_module.torch_chunk_gated_delta_rule
            original_l2norm = runtime_module.l2norm
            original_softplus = runtime_module.F.softplus
            l2_inputs = []
            softplus_inputs = []

            def capture_l2norm(value, *args, **kwargs):
                if len(l2_inputs) < 2:
                    l2_inputs.append(_torch_input_array(value))
                return original_l2norm(value, *args, **kwargs)

            def capture_softplus(value, *args, **kwargs):
                if not softplus_inputs:
                    softplus_inputs.append(_torch_input_array(value))
                return original_softplus(value, *args, **kwargs)

            def capture_rule(query, key, value, *args, **kwargs):
                result = original_rule(query, key, value, *args, **kwargs)
                g = kwargs.get('g', args[0] if len(args) > 0 else None)
                beta = kwargs.get('beta', args[1] if len(args) > 1 else None)
                tensors = {
                    'query': query,
                    'key': key,
                    'value': value,
                    'g': g,
                    'beta': beta,
                    'core_output': _first_torch_tensor(result),
                }
                required = set(tensors)
                if any(tensor is None for tensor in tensors.values()) or len(l2_inputs) != 2 or len(softplus_inputs) != 1:
                    raise RuntimeError(
                        'GDN recurrence receipt requires two L2 inputs, one softplus input, and '
                        f'exactly {sorted(required)}'
                    )
                converted = {name: _torch_input_array(tensor) for name, tensor in tensors.items()}
                converted['query_pre_norm'], converted['key_pre_norm'] = l2_inputs
                converted['softplus_input'] = softplus_inputs[0]
                self._internal_boundary_calls.append(converted)
                _write_named_internal_receipt(
                    self.output_dir,
                    'torch',
                    rank,
                    self.internal_boundary,
                    module_name,
                    module_class,
                    self._internal_boundary_calls,
                )
                return result

            setattr(runtime_module, 'l2norm', capture_l2norm)
            setattr(runtime_module.F, 'softplus', capture_softplus)
            setattr(runtime_module, 'torch_chunk_gated_delta_rule', capture_rule)

            def restore_recurrence_receipt():
                setattr(runtime_module, 'l2norm', original_l2norm)
                setattr(runtime_module.F, 'softplus', original_softplus)
                setattr(runtime_module, 'torch_chunk_gated_delta_rule', original_rule)

            self._internal_boundary_restore = restore_recurrence_receipt
            return

        def capture(_module, inputs, kwargs, output):
            input_tensor = _first_torch_call_tensor(inputs, kwargs)
            output_tensor = _first_torch_tensor(output)
            if input_tensor is None or output_tensor is None:
                raise RuntimeError(f'{self.internal_boundary} receipt requires tensor input and output')
            self._internal_boundary_calls.append({
                'input': _torch_input_array(input_tensor),
                'output': _torch_input_array(output_tensor),
            })
            _write_internal_boundary_receipt(
                self.output_dir, rank, self.internal_boundary, module_name, module_class, self._internal_boundary_calls
            )

        self._internal_boundary_handle = module.register_forward_hook(capture, with_kwargs=True)

    def _close_internal_boundary_receipt(self):
        if self._internal_boundary_handle is not None:
            self._internal_boundary_handle.remove()
            self._internal_boundary_handle = None
        if self._internal_boundary_restore is not None:
            self._internal_boundary_restore()
            self._internal_boundary_restore = None
        if self.internal_boundary and not self._internal_boundary_calls:
            raise RuntimeError(f'internal boundary {self.internal_boundary!r} produced no first-step calls')
        if self.internal_boundary == 'language_layer0_gdn_recurrence' and len(self._internal_boundary_calls) != 1:
            raise RuntimeError(
                'GDN recurrence receipt requires exactly one first-step call, '
                f'got {len(self._internal_boundary_calls)}'
            )

    def on_model_inputs(self, inputs, labels=None, phase=None):
        if not self.capture_model_inputs or self._model_inputs_captured:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _write_model_input_receipt(self.output_dir, 'torch', rank, inputs, labels, self.state.iteration, phase)
        self._model_inputs_captured = True

    def on_log(self, logs):
        if 'loss' not in logs:
            return
        self._close_internal_boundary_receipt()
        if not self.is_write_rank:
            return
        loss = logs['loss']
        if not isinstance(loss, (float, int)):
            raise TypeError(f'expected scalar Python loss after trainer normalization, got {type(loss)}')
        self.losses_by_iteration[int(self.state.iteration)] = float(loss)
        self._write_losses()
        metrics_path = self.output_dir / 'repro_metrics.jsonl'
        record = {'iteration': int(self.state.iteration), **logs}
        with metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()

    def on_save(self, output_dir):
        if not self.is_write_rank:
            return
        source = Path(output_dir)
        tensor_files = sorted(source.glob('*.safetensors'))
        if not tensor_files:
            raise FileNotFoundError(f'no safetensors produced in {source}')
        target = self.output_dir / 'checkpoint'
        temp = self.output_dir / f'.checkpoint.tmp-{os.getpid()}'
        shutil.rmtree(temp, ignore_errors=True)
        temp.mkdir(parents=True)
        copied = []
        for source_file in tensor_files:
            if source_file.is_symlink() or not source_file.is_file():
                raise ValueError(f'checkpoint source must be a regular file: {source_file}')
            target_file = temp / source_file.name
            shutil.copy2(source_file, target_file)
            copied.append(
                {
                    'path': target_file.name,
                    'bytes': target_file.stat().st_size,
                    'sha256': _sha256_file(target_file),
                    'nlink': target_file.stat().st_nlink,
                }
            )
        for name in ('model.safetensors.index.json', 'config.json', 'repro_profile.json'):
            source_file = source / name
            if source_file.is_file() and not source_file.is_symlink():
                shutil.copy2(source_file, temp / name)
        _atomic_json(temp / 'checkpoint_manifest.json', {'source': str(source), 'files': copied})
        if target.exists():
            shutil.rmtree(target)
        os.replace(temp, target)

    def on_train_end(self):
        self._close_internal_boundary_receipt()
        if not self.is_write_rank:
            return
        self._write_losses()
        if len(self.losses_by_iteration) != self.args.train_iters:
            raise RuntimeError(
                f'repro receipt expected {self.args.train_iters} logged losses, got {len(self.losses_by_iteration)}')
