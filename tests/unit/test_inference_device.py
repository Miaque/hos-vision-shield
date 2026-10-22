"""设备选择不改变检测策略，显式 GPU 不可用时不能降级。"""

import pytest
from pydantic import ValidationError

from vision_shield.config import Settings


def test_device_is_independent_and_cpu_backend_rejects_cuda():
    assert Settings(_env_file=None).inference_device == "cpu"
    assert Settings(_env_file=None, inference_device="cuda").detection_backend == "quality"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, detection_backend="cpu", inference_device="cuda")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, inference_device="auto")


@pytest.mark.parametrize("backend", ["quality", "quality_yolo", "egoblur"])
def test_factory_uses_cuda_for_primary_and_quality_detail(monkeypatch, backend):
    from types import SimpleNamespace
    from vision_shield.detection import load_detector, quality, egoblur, cuda_detail, yolo

    primary, detail = [], []
    def primary_model(*args, **kwargs):
        primary.append(kwargs)
        return SimpleNamespace()
    def detail_model(*args, **kwargs):
        detail.append(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(quality, "PrimaryDetector", primary_model)
    monkeypatch.setattr(quality, "DetailDetector", lambda *a, **kw: pytest.fail("CUDA 不得创建 CPU 补充模型"))
    monkeypatch.setattr(cuda_detail, "Detector", detail_model)
    monkeypatch.setattr(egoblur, "Detector", lambda *a, **kw: primary.append(kw))
    monkeypatch.setattr(yolo, "Detector", primary_model)
    detector = load_detector(Settings(_env_file=None, detection_backend=backend, inference_device="cuda").model_dump())
    assert primary[0]["device"] == "cuda"
    assert len(detail) == (1 if backend in {"quality", "quality_yolo"} else 0)
    if backend in {"quality", "quality_yolo"}:
        assert detector.primary.stage_times is detector.detail.stage_times is detector.stage_times


def test_unavailable_cuda_fails_before_loading_models(monkeypatch):
    from vision_shield.detection import egoblur

    monkeypatch.setattr(egoblur.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        egoblur.Detector("absent-face", "absent-plate", device="cuda")


def test_primary_captures_selected_cuda_ordinal_before_starting_model_threads(monkeypatch):
    from vision_shield.detection import egoblur

    devices, streams = [], []

    class Model:
        def to(self, device):
            devices.append(device)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(egoblur.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(egoblur.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(egoblur.torch.cuda, "Stream", lambda *, device: streams.append(device))
    monkeypatch.setattr(egoblur.torch.jit, "load", lambda *a, **kw: Model())
    monkeypatch.setattr(egoblur.torch, "set_num_threads", lambda value: None)
    detector = egoblur.Detector("face", "plate", device="cuda")
    assert detector.device == egoblur.torch.device("cuda", 2)
    assert devices == [detector.device, detector.device] and streams == [detector.device]
