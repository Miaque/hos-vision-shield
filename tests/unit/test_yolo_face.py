from types import SimpleNamespace

import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.detection import Detection


def test_yolo_quality_requires_cuda():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, detection_backend="quality_yolo", inference_device="cpu")
    assert Settings(_env_file=None, detection_backend="quality_yolo", inference_device="cuda").detection_backend == "quality_yolo"


def detector(monkeypatch, output):
    from vision_shield.detection import egoblur, yolo

    instance = yolo.Detector.__new__(yolo.Detector)
    instance.max_side, instance.face_threshold = 1280, 0.1
    inputs = []

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, names, values):
            inputs.append(values["image"])
            return [output]

    instance._session = lambda height, width: Session()
    plates = []
    def plate(self, rgb, *, kind=None):
        plates.append(kind)
        return [Detection("plate", (200, 50, 250, 70), 0.8)]
    monkeypatch.setattr(egoblur.Detector, "candidates", plate)
    return instance, inputs, plates


def test_face_decode_restores_source_padding_and_deduplicates(monkeypatch):
    output = np.zeros((1, 20, 2), np.float32)
    output[0, :4, :] = np.array([[64, 40 * 320 / 300 + 10, 64, 40 * 320 / 300]]).T
    output[0, 4] = [0.8, 0.7]
    instance, inputs, plates = detector(monkeypatch, output)
    result = instance.candidates(np.full((100, 300, 3), 255, np.uint8))
    assert len(result) == 2 and [d.kind for d in result] == ["face", "plate"]
    np.testing.assert_allclose(result[0].box, (30, 20, 90, 60), atol=0.001)
    assert inputs[0].shape == (1, 3, 128, 320) and plates == ["plate"]
    assert (inputs[0][:, :, 10:117] == 1).all()


@pytest.mark.parametrize("fault", ["shape", "nan", "score", "extent"])
def test_invalid_face_outputs_fail_instead_of_returning_plates(monkeypatch, fault):
    output = np.zeros((1, 20, 1), np.float32)
    output[0, :5, 0] = [20, 20, 10, 10, 0.8]
    if fault == "shape":
        output = output[:, :5]
    elif fault == "nan":
        output[0, 0, 0] = np.nan
    elif fault == "score":
        output[0, 4, 0] = 1.1
    else:
        output[0, 2, 0] = -1
    instance, _, plates = detector(monkeypatch, output)
    with pytest.raises(ValueError, match="Invalid face detector output"):
        instance.candidates(np.zeros((100, 300, 3), np.uint8))
    assert not plates


def test_plate_confirmation_does_not_run_face_session(monkeypatch):
    instance, inputs, plates = detector(monkeypatch, None)
    assert instance.candidates(np.zeros((100, 300, 3), np.uint8), kind="plate")[0].kind == "plate"
    assert inputs == [] and plates == ["plate"]


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_session_cache_is_bounded_and_rejects_cpu_fallback(monkeypatch, tmp_path, provider):
    import sys
    from vision_shield.detection import yolo

    created = []
    class Session:
        def __init__(self, *args, **kwargs):
            created.append(self)

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_providers(self):
            return [provider]

    class Options:
        def add_free_dimension_override_by_name(self, *args):
            pass

        def add_session_config_entry(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session, SessionOptions=Options))
    monkeypatch.setattr(yolo.torch.cuda, "current_device", lambda: 0)
    instance = yolo.Detector.__new__(yolo.Detector)
    instance.face_path, instance.sessions = tmp_path / "face.onnx", {}
    if provider == "CPUExecutionProvider":
        with pytest.raises(RuntimeError, match="CUDA provider"):
            instance._session(160, 160)
        assert not instance.sessions
        return
    first = instance._session(160, 160)
    assert instance._session(160, 160) is first and len(created) == 1
    for side in (192, 224, 256, 288):
        instance._session(side, side)
    assert len(instance.sessions) == 4 and (160, 160) not in instance.sessions
    assert all(s.fallback_disabled for s in created)


def test_empty_faces_and_outside_boxes_never_mask_unrelated_pixels(monkeypatch):
    output = np.zeros((1, 20, 2), np.float32)
    output[0, :5] = np.array([[-200, -200, 10, 10, 0.8], [0, 0, 0, 0, 0]]).T
    instance, _, _ = detector(monkeypatch, output)
    assert instance.candidates(np.zeros((100, 300, 3), np.uint8), kind="face") == []
    output[0, 4] = 0
    assert instance.candidates(np.zeros((100, 300, 3), np.uint8), kind="face") == []
