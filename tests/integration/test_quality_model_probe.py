"""A bounded sample review, never a claim of dataset-wide recall."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_shield.config import Settings
from vision_shield.detection import load_detector
from vision_shield.redaction import redact


def test_quality_restores_distant_plate_and_removes_known_large_false_masks():
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    detector = load_detector(Settings(_env_file=None, detection_backend="quality").model_dump())
    for case in manifest["cases"]:
        source = Path(manifest["local_source"]) / case["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        with Image.open(source) as image:
            rgb = np.array(image.convert("RGB"))
        masked = redact(rgb, detector.detect(rgb))
        for box in case["masked_regions"]:
            left, top, right, bottom = box
            assert not masked[top:bottom, left:right].any(), (case["id"], box)
        for box in case["preserved_regions"]:
            left, top, right, bottom = box
            assert np.array_equal(masked[top:bottom, left:right], rgb[top:bottom, left:right]), (case["id"], box)


@pytest.mark.parametrize("case_id", ["video-04-frame-09", "video-04-frame-10", "video-04-frame-16",
                                   "video-04-frame-19", "video-04-frame-24", "image-7"])
def test_quality_rechecks_weak_support_without_losing_faces_or_masking_wheel(case_id):
    """去除轮胎误遮时，同帧车牌和依赖弱佐证的真实侧脸仍须遮挡。"""
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == case_id)
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    with Image.open(source) as image:
        rgb = np.array(image.convert("RGB"))
    detector = load_detector(Settings(_env_file=None, detection_backend="quality").model_dump())
    masked = redact(rgb, detector.detect(rgb))
    for left, top, right, bottom in case["masked_regions"]:
        assert not masked[top:bottom, left:right].any(), (case_id, "真实目标必须遮挡")
    for left, top, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[top:bottom, left:right], rgb[top:bottom, left:right]), (case_id, "误遮区域必须保留原像素")
