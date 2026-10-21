"""显式启用的真实 CUDA 验证；不能以 CPU 回退冒充 GPU 通过。"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vision_shield.config import Settings
from vision_shield.detection import load_detector
from vision_shield.detection.preflight import run_preflight
from vision_shield.redaction import redact


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="设置 RUN_GPU_TESTS=1 才运行真实 GPU 验证")
@pytest.mark.parametrize("case_id", ["image-7", "face-video2-10s"])
def test_quality_cuda_runs_local_models_and_preserves_reviewed_regions(case_id):
    assert torch.cuda.is_available(), "显式 GPU 验证要求 CUDA 可用"
    config = Settings(_env_file=None, inference_device="cuda")
    report = run_preflight(config.model_dump(mode="json"))
    assert len(report["models"]) == 4
    assert report["model_devices"] == ["cuda"] and report["nms_device"] == "cuda"
    assert report["detail_providers"] == ["CUDAExecutionProvider"] * 2
    assert report["cuda_build"] and report["gpu_name"] and report["architectures"]
    detector = load_detector(config.model_dump())
    assert detector.primary.device.type == "cuda"
    assert all(next(model.parameters()).is_cuda for model in detector.primary.models)
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == case_id)
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    with Image.open(source) as image:
        pixels = np.array(image.convert("RGB"))
    assert detector.detail.parallel is True
    detector.detail.parallel = False
    expected = detector.detect(pixels)
    detector.detail.parallel = True
    actual = detector.detect(pixels)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.001,
                               err_msg="并行补充检测仅允许亚像素浮点波动，遮挡像素须完全相同")
    masked = redact(pixels, actual)
    assert np.array_equal(masked, redact(pixels, expected))
    for left, top, right, bottom in case["masked_regions"]:
        assert (masked[top:bottom, left:right] == 64).all(), "已标记目标必须遮挡"
    for left, top, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[top:bottom, left:right], pixels[top:bottom, left:right]), "误遮区域应保留"


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="设置 RUN_GPU_TESTS=1 才运行真实 GPU 验证")
def test_detail_model_nodes_execute_on_cuda(monkeypatch, tmp_path):
    import onnxruntime as ort
    from vision_shield.detection.cuda_detail import Detector

    create = ort.InferenceSession
    sessions = []

    def profiled(*args, **kwargs):
        options = kwargs["sess_options"]
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / f"detail-{len(sessions)}")
        session = create(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(ort, "InferenceSession", profiled)
    detector = Detector("models/yunet_face.onnx", "models/hyperlpr_plate.onnx")
    detector.candidates(np.zeros((320, 320, 3), np.uint8))
    assert len(sessions) == 2
    for session in sessions:
        events = json.loads(Path(session.end_profiling()).read_text("utf-8"))
        providers = {event["args"]["provider"] for event in events if "provider" in event.get("args", {})}
        assert providers == {"CUDAExecutionProvider"}, "不能仅检查已安装 provider，必须核对实际算子"


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="设置 RUN_GPU_TESTS=1 才运行真实 GPU 验证")
@pytest.mark.parametrize("case_id", ["image-7", "face-video2-10s", "video-04-frame-24"])
def test_yolo_quality_keeps_reviewed_regions_and_all_face_nodes_on_cuda(monkeypatch, tmp_path, case_id):
    import onnxruntime as ort

    create, sessions = ort.InferenceSession, []
    def profiled(path, *args, **kwargs):
        if str(path).endswith("yolov8n_face.onnx"):
            options = kwargs["sess_options"]
            options.enable_profiling = True
            options.profile_file_prefix = str(tmp_path / f"face-{len(sessions)}")
        session = create(path, *args, **kwargs)
        if str(path).endswith("yolov8n_face.onnx"):
            sessions.append(session)
        return session

    monkeypatch.setattr(ort, "InferenceSession", profiled)
    config = Settings(_env_file=None, inference_device="cuda", detection_backend="quality_yolo")
    report = run_preflight(config.model_dump(mode="json"))
    assert len(report["models"]) == 4 and report["primary_face_providers"] == ["CUDAExecutionProvider"]
    detector = load_detector(config.model_dump())
    assert detector.primary.models[0] is None and next(detector.primary.models[1].parameters()).is_cuda
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    path = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    pixels = np.array(Image.open(path).convert("RGB"))
    masked = redact(pixels, detector.detect(pixels))
    for x, y, right, bottom in case["masked_regions"]:
        assert (masked[y:bottom, x:right] == 64).all()
    for x, y, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[y:bottom, x:right], pixels[y:bottom, x:right])
    providers = set()
    for session in sessions:
        events = json.loads(Path(session.end_profiling()).read_text("utf-8"))
        providers.update(event["args"]["provider"] for event in events if "provider" in event.get("args", {}))
    assert providers == {"CUDAExecutionProvider"}, "主人脸网络也必须实际全CUDA执行"
