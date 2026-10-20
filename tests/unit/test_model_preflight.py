"""离线模型身份、加载和设备错误均须在就绪前明确失败。"""

import hashlib
import json

import pytest

from vision_shield.config import Settings
from vision_shield.detection import preflight


@pytest.fixture
def model_config(tmp_path):
    names = ("face.jit", "license_plate.jit", "yunet_face.onnx", "hyperlpr_plate.onnx")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    for manifest, selected in [("manifest.json", names[:2]), ("cpu-manifest.json", names[2:])]:
        (tmp_path / manifest).write_text(json.dumps({"models": [dict(path=name, sha256=hashlib.sha256(name.encode()).hexdigest())
                                                               for name in selected]}), encoding="utf-8")
    return Settings(_env_file=None, face_model_path=tmp_path / names[2], plate_model_path=tmp_path / names[3],
                    egoblur_face_model_path=tmp_path / names[0], egoblur_plate_model_path=tmp_path / names[1]).model_dump(mode="json")


@pytest.mark.parametrize("backend,count", [("quality", 4), ("cpu", 2), ("egoblur", 2)])
def test_only_selected_roles_are_verified(model_config, backend, count):
    model_config["detection_backend"] = backend
    if backend == "egoblur":
        model_config["face_model_path"] = model_config["egoblur_face_model_path"]
        model_config["plate_model_path"] = model_config["egoblur_plate_model_path"]
    assert len(preflight.verify_models(model_config)) == count


@pytest.mark.parametrize("fault", ["missing", "corrupt", "unreadable", "manifest"])
def test_bad_model_never_reaches_loading(model_config, monkeypatch, fault):
    from pathlib import Path
    path = Path(model_config["egoblur_face_model_path"])
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"wrong")
    elif fault == "manifest":
        (path.parent / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        original = Path.open
        def denied(self, *args, **kwargs):
            if self == path:
                raise PermissionError("凭据不得进入诊断")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "open", denied)
    monkeypatch.setattr(preflight, "run_inference", lambda cfg: pytest.fail("坏模型不得加载"))
    with pytest.raises(preflight.PreflightError) as error:
        preflight.run_preflight(model_config)
    assert "face.jit" in str(error.value) and "凭据" not in str(error.value)


def test_inference_exception_is_bounded_diagnostic(model_config, monkeypatch):
    def fails(cfg):
        raise RuntimeError("rtsp://secret:password@camera/private")
    monkeypatch.setattr(preflight, "run_inference", fails)
    with pytest.raises(preflight.PreflightError, match="加载与推理") as error:
        preflight.run_preflight(model_config)
    assert "password" not in str(error.value)


@pytest.mark.parametrize("identifier", ["", "all", "0,1", "-1", "GPU-short", " 0"])
def test_invalid_deployment_gpu_selection_fails(monkeypatch, identifier):
    monkeypatch.setenv("GPU_DEVICE_ID", identifier)
    with pytest.raises(preflight.PreflightError, match="GPU_DEVICE_ID"):
        preflight.validate_device("cuda")


@pytest.mark.parametrize("identifier", ["0", "3", "GPU-12345678-1234-1234-1234-123456789abc"])
def test_deployment_gpu_requires_one_visible_card(monkeypatch, identifier):
    import torch
    monkeypatch.setenv("GPU_DEVICE_ID", identifier)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(preflight.PreflightError, match="唯一"):
        preflight.validate_device("cuda")


def test_native_cuda_does_not_gain_single_card_restriction(monkeypatch):
    import torch
    monkeypatch.delenv("GPU_DEVICE_ID", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    preflight.validate_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(preflight.PreflightError, match="CUDA"):
        preflight.validate_device("cuda")


def test_single_visible_deployment_gpu_and_nms_failure(model_config, monkeypatch):
    import torch
    import torchvision
    from types import SimpleNamespace
    from vision_shield import detection
    monkeypatch.setenv("GPU_DEVICE_ID", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    preflight.validate_device("cuda")
    detector = SimpleNamespace(models=[torch.nn.Linear(1, 1)], detect=lambda frame: [])
    monkeypatch.setattr(detection, "load_detector", lambda cfg: detector)
    def fails(*args):
        raise RuntimeError("NMS 内部凭据")
    monkeypatch.setattr(torchvision.ops, "nms", fails)
    with pytest.raises(preflight.PreflightError, match="加载与推理") as error:
        preflight.run_preflight(model_config)
    assert "凭据" not in str(error.value)


def test_cuda_probe_rejects_model_left_on_cpu(model_config, monkeypatch):
    import torch
    from types import SimpleNamespace
    from vision_shield import detection
    monkeypatch.setenv("GPU_DEVICE_ID", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(detection, "load_detector", lambda cfg: SimpleNamespace(models=[torch.nn.Linear(1, 1)], detect=lambda frame: []))
    model_config["inference_device"] = "cuda"
    with pytest.raises(preflight.PreflightError, match="模型与请求设备"):
        preflight.run_preflight(model_config)
