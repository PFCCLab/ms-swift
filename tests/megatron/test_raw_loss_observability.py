from swift.megatron.callbacks.print import raw_loss_event
from swift.megatron.trainers.trainer import project_owning_loader_semantics


def test_raw_loss_event_preserves_unrounded_values_and_step():
    logs = {
        'loss': 12.410510059999999,
        'mtp_0_loss': 13.367947579999999,
        'eval_loss': 99.0,
    }
    assert raw_loss_event(1, logs) == {
        'step': 1,
        'loss': 12.410510059999999,
        'mtp_0_loss': 13.367947579999999,
    }


def test_raw_loss_event_omits_non_loss_metrics():
    assert raw_loss_event(3, {'grad_norm': 1.0, 'learning_rate': 1e-6}) is None


def test_owning_loader_projection_removes_sp_padding_and_reverses_label_roll():
    input_values = list(range(57)) + [154820]
    original_labels = [-100] * 13 + list(range(44)) + [-100]
    shifted_labels = original_labels[1:] + original_labels[:1]
    semantic_input, semantic_labels, semantic_mask = project_owning_loader_semantics(
        input_values, shifted_labels, semantic_length=57, labels_were_shifted=True)
    assert semantic_input == list(range(57))
    assert semantic_labels == original_labels[:57]
    assert sum(semantic_mask) == 44


def test_owning_loader_projection_rejects_invalid_semantic_length():
    try:
        project_owning_loader_semantics([1, 2], [-100, 2], semantic_length=3)
    except ValueError as exc:
        assert 'invalid owning-loader semantic length' in str(exc)
    else:
        raise AssertionError('invalid semantic length was accepted')
