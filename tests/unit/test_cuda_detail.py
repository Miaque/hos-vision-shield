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
