import json
from types import SimpleNamespace

from swift.megatron.callbacks import repro_receipt


def test_repro_receipt_preserves_unrounded_loss_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_receipt, 'is_last_rank', lambda: True)
    state = SimpleNamespace(iteration=1)
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path), train_iters=2), state=state)
    callback = repro_receipt.ReproReceiptCallback(trainer)

    callback.on_log({'loss': 9.107421875})
    state.iteration = 2
    callback.on_log({'loss': 8.9931640625})
    callback.on_train_end()

    receipt = json.loads((tmp_path / 'loss.json').read_text())
    assert receipt == {'framework': 'torch', 'losses': [9.107421875, 8.9931640625]}
    metrics = (tmp_path / 'repro_metrics.jsonl').read_text().splitlines()
    assert len(metrics) == 2


def test_repro_receipt_copies_regular_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(repro_receipt, 'is_last_rank', lambda: True)
    state = SimpleNamespace(iteration=1)
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path), train_iters=1), state=state)
    callback = repro_receipt.ReproReceiptCallback(trainer)
    source = tmp_path / 'checkpoint-1'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'fixture')

    callback.on_save(str(source))

    copied = tmp_path / 'checkpoint' / 'model.safetensors'
    assert copied.read_bytes() == b'fixture'
    assert not copied.is_symlink()
    assert copied.stat().st_nlink == 1


def test_internal_boundary_receipt_preserves_call_order_and_raw_bits(tmp_path):
    import numpy as np
    import torch

    calls = []
    for offset in (0, 4):
        input_tensor = torch.arange(offset, offset + 4, dtype=torch.bfloat16).reshape(2, 2)
        output_tensor = input_tensor + 1
        calls.append({
            'input': repro_receipt._torch_input_array(input_tensor),
            'output': repro_receipt._torch_input_array(output_tensor),
        })

    repro_receipt._write_internal_boundary_receipt(
        tmp_path, 0, 'vision_merger', 'visual.merger', 'example.Merger', calls
    )

    manifest = json.loads((tmp_path / 'internal_boundaries' / 'vision_merger_rank0.json').read_text())
    arrays = np.load(tmp_path / 'internal_boundaries' / manifest['npz'])
    assert manifest['call_count'] == 2
    assert [call['call_index'] for call in manifest['calls']] == [0, 1]
    assert manifest['calls'][0]['input']['dtype'] == 'bfloat16'
    assert arrays['c0_input'].dtype == np.uint16


def test_internal_boundary_tensor_selection_supports_keyword_hidden_states():
    import torch

    hidden_states = torch.ones(2, 3, dtype=torch.bfloat16)
    attention_mask = torch.zeros(2, 3, dtype=torch.bool)
    assert repro_receipt._first_torch_call_tensor((), {
        'attention_mask': attention_mask,
        'hidden_states': hidden_states,
    }) is hidden_states
    assert repro_receipt._first_torch_call_tensor((hidden_states,), {}) is hidden_states
