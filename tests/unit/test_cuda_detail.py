import numpy as np
import pytest


def test_yunet_decode_keeps_stride_coordinates_and_score():
    from vision_shield.detection.cuda_detail import decode_faces

    outputs = {}
    for stride in (8, 16, 32):
        count = (32 // stride) ** 2
        for name, size in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
            outputs[f"{name}_{stride}"] = np.zeros((1, count, size), np.float32)
    outputs["cls_8"][0, 5, 0] = 0.81
    outputs["obj_8"][0, 5, 0] = 1
    faces = decode_faces(outputs, 32, 32, 0.3)
    assert faces.shape == (1, 15)
    np.testing.assert_allclose(faces[0, :4], [4, 4, 8, 8])
    assert faces[0, -1] == pytest.approx(0.9)
    outputs["cls_8"][0, 5, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid face"):
        decode_faces(outputs, 32, 32, 0.3)


def test_cuda_detail_refuses_cpu_only_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="require ONNX Runtime CUDA"):
        Detector("unused-face", "unused-plate")


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_fixed_face_sessions_keep_dimensions_device_and_bounded_lifetime(monkeypatch, provider):
    import sys
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    created = []

    class Options:
        def __init__(self):
            self.dimensions, self.entries = {}, {}

        def add_free_dimension_override_by_name(self, name, value):
            self.dimensions[name] = value

        def add_session_config_entry(self, name, value):
            self.entries[name] = value

    class Session:
        def __init__(self, model, *, sess_options, providers):
            assert model == b"same-weights"
            assert providers[0][1]["device_id"] == 2
            assert sess_options.entries["session.disable_cpu_ep_fallback"] == "1"
            self.dimensions, self.disabled = sess_options.dimensions, False
            created.append(self)

        def disable_fallback(self):
            self.disabled = True

        def get_providers(self):
            return [provider]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=Options, InferenceSession=Session))
    detector = Detector.__new__(Detector)
    detector.device, detector.face_model, detector.face_sessions = torch.device("cuda", 2), b"same-weights", {}
    if provider != "CUDAExecutionProvider":
        with pytest.raises(RuntimeError, match="CUDA provider"):
            detector._face_session(384, 1280)
        assert not detector.face_sessions
        return
    first = detector._face_session(384, 1280)
    assert detector._face_session(384, 1280) is first
    assert first.dimensions == {"height": 384, "width": 1280}
    for height in (416, 448, 480, 512):
        detector._face_session(height, 1280)
    assert len(detector.face_sessions) == 4 and (384, 1280) not in detector.face_sessions
    assert len(created) == 5 and all(session.disabled for session in created)
