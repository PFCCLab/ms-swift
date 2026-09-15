# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import os
import torch
import torch.distributed as dist
import torch.nn
from collections import defaultdict
from functools import partial
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional

from swift.utils import get_logger
from ..init import _use_accuracy_compatible_enabled
from .base import BaseMegatronTrainer

logger = get_logger()


def project_owning_loader_semantics(input_values, model_label_values, semantic_length, labels_were_shifted=True):
    """Normalize padded Megatron carrier tensors back to the dataset semantic row."""
    semantic_length = int(semantic_length)
    if semantic_length <= 0 or semantic_length > len(input_values) or semantic_length > len(model_label_values):
        raise ValueError(f'invalid owning-loader semantic length {semantic_length} for carrier lengths '
                         f'{len(input_values)}/{len(model_label_values)}')
    semantic_input_values = input_values[:semantic_length]
    normalized_label_values = model_label_values
    if labels_were_shifted and model_label_values:
        # get_batch_on_this_pp_rank rolls causal-LM labels left by one before
        # model forward. Reverse that roll for a framework-neutral dataset receipt.
        normalized_label_values = model_label_values[-1:] + model_label_values[:-1]
    semantic_label_values = normalized_label_values[:semantic_length]
    semantic_mask_values = [label != -100 for label in semantic_label_values]
    return semantic_input_values, semantic_label_values, semantic_mask_values


class MegatronTrainer(BaseMegatronTrainer):

    def _write_input_contract_once(self, data, seq_lens=None):
        path = os.environ.get('MODEL_REPRO_INPUT_RECEIPT_PATH')
        if not path or getattr(self, '_input_contract_written', False):
            return
        if not mpu.is_pipeline_last_stage(ignore_virtual=False):
            return
        if (torch.distributed.is_initialized()
                and torch.distributed.get_rank() != torch.distributed.get_world_size() - 1):
            return
        input_ids = data.get('input_ids')
        labels = data.get('labels')
        if input_ids is None or labels is None:
            return

        def values(tensor):
            return tensor.detach().to(device='cpu', dtype=torch.int64).reshape(-1).tolist()

        def digest(items):
            return hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()

        input_values = values(input_ids)
        label_values = values(labels)
        model_mask_values = [label != -100 for label in label_values]
        semantic_length = seq_lens[0] if seq_lens else len(input_values)
        labels_were_shifted = self.args.task_type == 'causal_lm'
        semantic_input_values, semantic_label_values, semantic_mask_values = project_owning_loader_semantics(
            input_values, label_values, semantic_length, labels_were_shifted)
        payload = {
            'schema': 'glm52-owning-loader-input/v1',
            'framework': 'torch',
            'rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            'step': self.state.iteration + 1,
            'input_ids': {
                'shape': list(input_ids.shape),
                'dtype': str(input_ids.dtype),
                'count': len(input_values),
                'sha256': digest(input_values),
            },
            'labels': {
                'shape': list(labels.shape),
                'dtype': str(labels.dtype),
                'count': len(label_values),
                'supervised_count': sum(model_mask_values),
                'sha256': digest(label_values),
                'projection': 'model_next_token_labels',
            },
            'loss_mask': {
                'shape': list(labels.shape),
                'dtype': 'bool',
                'count': len(model_mask_values),
                'supervised_count': sum(model_mask_values),
                'sha256': digest(model_mask_values),
            },
            'semantic': {
                'input_token_count': len(semantic_input_values),
                'supervised_target_count': sum(semantic_mask_values),
                'input_ids_sha256': digest(semantic_input_values),
                'labels_sha256': digest(semantic_label_values),
                'loss_mask_sha256': digest(semantic_mask_values),
                'projection': 'dataset_row_before_megatron_padding_and_label_roll',
            },
            'carrier_padding': {
                'count': len(input_values) - len(semantic_input_values),
                'input_ids_sha256': digest(input_values[len(semantic_input_values):]),
                'labels_sha256': digest(label_values[len(semantic_input_values):]),
            },
            'ignore_index': -100,
            'dataset': os.environ.get('MODEL_REPRO_INPUT_DATASET_PATH'),
        }
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        self._input_contract_written = True

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
        logits = self.get_last_tokens(output_tensor, packed_seq_params, attention_mask)
        num_labels = args.num_labels
        acc = None
        if args.problem_type == 'regression':
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif args.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            acc = (logits.detach().argmax(dim=-1) == labels).float().mean()
        elif args.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            preds = logits.sigmoid() > 0.5
            acc = (labels == preds).all(dim=-1).float().mean()
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None):
        args = self.args

        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        masked_losses = losses * loss_mask
        if _use_accuracy_compatible_enabled() and self.config.accuracy_compatible_loss_sum_dtype == 'float64':
            loss_sum = masked_losses.reshape(-1).double().sum().float()
        else:
            loss_sum = torch.sum(masked_losses)
        loss = torch.cat([loss_sum.view(1), loss_mask.sum().view(1)])

        # Reduce loss for logging.
        reporting_loss = loss.detach().clone()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group(with_context_parallel=True))

        lm_loss = loss[0]
        lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)

        if _use_accuracy_compatible_enabled():
            # 精度对齐锚点 2（对应 PF language_loss.py forward_impl 出口的 final_loss）：
            # 本 rank、本 micro-batch 的 sum(loss*mask)/valid_tokens，
            # 未跨 DP all-reduce、未除 num_microbatches，与 PF 侧同语义。
            import hashlib as _hashlib
            _final = (loss[0].detach().float() / loss[1].detach().float().clamp(min=1)).contiguous()
            print(
                f"\nfinal_loss: rank={torch.distributed.get_rank()} "
                f"val={_final.item():.20f} "
                f"md5={_hashlib.md5(_final.cpu().numpy().tobytes()).hexdigest()}",
                flush=True)

        metrics = {'loss': reporting_loss}
        if args.enable_channel_loss:
            metrics.update(self._compute_channel_loss(losses, loss_mask, channels, packed_seq_params))
        return (lm_loss, local_num_tokens, metrics)

    def _compute_channel_loss(self, losses, loss_mask, channels, packed_seq_params=None):
        args = self.args
        metrics = defaultdict(lambda: torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device()))
        if args.padding_free:
            num_samples = packed_seq_params.seq_lens.shape[0]
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            for i in range(cu_seqlens.shape[0] - 1):
                channel = None if channels is None else channels[i]
                slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                c_loss = losses[0, slice_][loss_mask[0, slice_]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]
        else:
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                c_loss = losses[i][loss_mask[i]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]

        # Synchronize keys to avoid getting stuck.
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_keys = [None] * torch.distributed.get_world_size(group=dp_cp_group)
        dist.all_gather_object(all_keys, list(metrics.keys()), group=dp_cp_group)
        new_metrics = {}
        for key in sorted(set().union(*all_keys)):
            new_metrics[key] = metrics[key]
        new_metrics = self._all_reduce_metric(new_metrics, torch.distributed.ReduceOp.SUM, group=dp_cp_group)
        return new_metrics

    def forward_step(self, data_iterator, model):
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        seq_lens = data.pop('_model_repro_seq_lens', None)
        self._write_input_contract_once(data, seq_lens)
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        if self.args.task_type == 'seq_cls':
            loss_func = partial(
                self.seq_cls_loss_func,
                labels=labels,
                packed_seq_params=packed_seq_params,
                attention_mask=data.get('attention_mask')
                if data.get('attention_mask') is not None else data.get('attention_mask_2d'))
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params)
        return output_tensor, loss_func
