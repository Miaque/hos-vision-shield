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
def test_quality_cuda_runs_local_models_and_preserves_reviewed_regions():
    assert torch.cuda.is_available(), "显式 GPU 验证要求 CUDA 可用"
    config = Settings(_env_file=None, inference_device="cuda")
    report = run_preflight(config.model_dump(mode="json"))
    assert len(report["models"]) == 4
    assert report["model_devices"] == ["cuda"] and report["nms_device"] == "cuda"
    assert report["cuda_build"] and report["gpu_name"] and report["architectures"]
    detector = load_detector(config.model_dump())
    assert detector.primary.device.type == "cuda"
    assert all(next(model.parameters()).is_cuda for model in detector.primary.models)
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == "image-7")
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    with Image.open(source) as image:
        pixels = np.array(image.convert("RGB"))
    masked = redact(pixels, detector.detect(pixels))
    for left, top, right, bottom in case["masked_regions"]:
        assert not masked[top:bottom, left:right].any(), "已标记目标必须遮挡"
    for left, top, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[top:bottom, left:right], pixels[top:bottom, left:right]), "误遮区域应保留"
