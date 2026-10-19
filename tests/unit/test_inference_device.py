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


@pytest.mark.parametrize("backend", ["quality", "egoblur"])
def test_factory_passes_cuda_to_primary_only(monkeypatch, backend):
    from vision_shield.detection import load_detector, quality, egoblur

    primary, detail = [], []
    monkeypatch.setattr(quality, "PrimaryDetector", lambda *a, **kw: primary.append(kw))
    monkeypatch.setattr(quality, "DetailDetector", lambda *a, **kw: detail.append(kw))
    monkeypatch.setattr(egoblur, "Detector", lambda *a, **kw: primary.append(kw))
    load_detector(Settings(_env_file=None, detection_backend=backend, inference_device="cuda").model_dump())
    assert primary[0]["device"] == "cuda"
    assert all("device" not in item for item in detail)


def test_unavailable_cuda_fails_before_loading_models(monkeypatch):
    from vision_shield.detection import egoblur

    monkeypatch.setattr(egoblur.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        egoblur.Detector("absent-face", "absent-plate", device="cuda")
