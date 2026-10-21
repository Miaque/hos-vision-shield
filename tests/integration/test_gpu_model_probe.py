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
def test_half_backbone_keeps_heads_float_and_reviewed_regions(case_id):
    from vision_shield.detection.egoblur import _MixedPrecisionModel

    detector = load_detector(Settings(_env_file=None, inference_device="cuda").model_dump())
    # 即使测试机没有Tensor Core，也显式验证混合精度路径，不能以旧路径通过冒充。
    detector.primary.models = [model if isinstance(model, _MixedPrecisionModel) else _MixedPrecisionModel(model, kind)
                               for model, kind in zip(detector.primary.models, ("face", "plate"))]
    for model in detector.primary.models:
        assert next(model.model.backbone.parameters()).dtype == torch.float16
        assert next(model.model.roi_heads.parameters()).dtype == torch.float32
        assert next(model.model.proposal_generator.parameters()).dtype == torch.float32
        assert model.model.pixel_mean.dtype == torch.float32
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            boxes, _, scores, size = model(torch.zeros((3, 319, 321), dtype=torch.uint8, device="cuda"))
        assert boxes.dtype == scores.dtype == torch.float32
        assert size.device.type == "cpu" and size.tolist() == [319, 321]
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == case_id)
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    pixels = np.array(Image.open(source).convert("RGB"))
    masked = redact(pixels, detector.detect(pixels))
    for left, top, right, bottom in case["masked_regions"]:
        assert (masked[top:bottom, left:right] == 64).all()
    for left, top, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[top:bottom, left:right], pixels[top:bottom, left:right])
