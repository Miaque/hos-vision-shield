from types import SimpleNamespace

import cv2
import pytest

from vision_shield.media import attempts


@pytest.mark.parametrize('error', [RuntimeError('model execution failed'), cv2.error('model execution failed'),
                                 ValueError('invalid pixels')])
def test_model_execution_runtime_errors_require_reload_but_validation_does_not(error):
    from vision_shield.detection.quality import _timed_candidates

    def candidates(*args, **kwargs):
        raise error

    with pytest.raises(type(error)) as caught:
        _timed_candidates(SimpleNamespace(candidates=candidates), object())
    assert attempts.model_runtime_failure(caught.value) is (not isinstance(error, ValueError))


@pytest.mark.parametrize('name,expected', [('Fail', True), ('EPFail', True), ('EngineError', True),
                                        ('RuntimeException', True), ('InvalidArgument', False), ('NoSuchFile', False)])
def test_onnx_runtime_failures_are_distinct_from_input_and_model_errors(name, expected):
    error_type = type(name, (Exception,), {'__module__': 'onnxruntime.capi.onnxruntime_pybind11_state'})
    assert attempts.model_runtime_failure(error_type('private:secret')) is expected


def test_device_memory_fault_reloads_without_using_a_gpu():
    import torch

    assert attempts.model_runtime_failure(torch.OutOfMemoryError('CUDA allocation failed'))


def test_lost_device_retries_but_permanent_preflight_errors_do_not(monkeypatch):
    import torch
    from vision_shield.detection import preflight

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(preflight.DeviceUnavailableError) as caught:
        preflight.validate_device('cuda')
    assert attempts.model_runtime_failure(caught.value)
    with pytest.raises(preflight.PreflightError) as caught:
        preflight.verify_models(dict(detection_backend='invalid'))
    assert not attempts.model_runtime_failure(caught.value)


def test_native_external_library_fault_recycles_the_device_context():
    from av.error import ExternalError

    assert attempts.model_runtime_failure(ExternalError(1, 'external device failed'))


@pytest.mark.parametrize('error', [RuntimeError('application bug'), ValueError('bad config'),
                                 ConnectionResetError('source disconnected'), FileNotFoundError('weights')])
def test_unrelated_errors_are_not_model_failures(error):
    try:
        raise error
    except Exception as caught:
        assert not attempts.model_runtime_failure(caught)
