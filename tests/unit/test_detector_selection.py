import pytest
from pydantic import ValidationError

from vision_shield.config import Settings


def test_quality_defaults_keep_both_model_families_and_unknown_backends_fail():
    config = Settings(_env_file=None)
    assert config.detection_backend == "quality"
    assert config.face_model_path.name == "yunet_face.onnx"
    assert config.plate_model_path.name == "hyperlpr_plate.onnx"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, detection_backend="cuda")


def test_selecting_cpu_never_loads_the_legacy_torch_detector(monkeypatch):
    from vision_shield.detection import load_detector
    from vision_shield.detection import cpu

    received = []
    sentinel = object()
    monkeypatch.setattr(cpu, "Detector", lambda *args: received.append(args) or sentinel)
    config = Settings(_env_file=None, detection_backend="cpu")
    assert load_detector(config.model_dump()) is sentinel
    assert received == [(config.face_model_path, config.plate_model_path,
                         config.detection_max_side, config.detection_box_padding)]
