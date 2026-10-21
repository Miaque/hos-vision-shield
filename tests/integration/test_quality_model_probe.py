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


@pytest.mark.parametrize("region,masked", [([393, 182, 400, 204], True), ([380, 480, 490, 520], False)])
def test_side_face_is_masked_and_handbag_is_preserved(region, masked):
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == "face-video2-10s")
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    with Image.open(source) as image:
        rgb = np.array(image.convert("RGB"))
    detector = load_detector(Settings(_env_file=None).model_dump())
    actual = redact(rgb, detector.detect(rgb))
    x1, y1, x2, y2 = region
    if masked:
        assert (actual[y1:y2, x1:x2] == 64).all(), "已确认的近处侧脸必须遮挡"
    else:
        assert np.array_equal(actual[y1:y2, x1:x2], rgb[y1:y2, x1:x2]), "手提袋不应被当作车牌遮挡"


def test_early_small_face_is_confirmed_before_tracking_can_help():
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == "face-video1-2s")
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    rgb = np.array(Image.open(source).convert("RGB"))
    detector = load_detector(Settings(_env_file=None).model_dump())
    actual = redact(rgb, detector.detect(rgb))
    for x1, y1, x2, y2 in case["masked_regions"]:
        assert (actual[y1:y2, x1:x2] == 64).all(), "已有弱小脸候选须先放大复核，不能等待后续高分帧"


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
            assert (masked[top:bottom, left:right] == 64).all(), (case["id"], box)
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
        assert (masked[top:bottom, left:right] == 64).all(), (case_id, "真实目标必须遮挡")
    for left, top, right, bottom in case["preserved_regions"]:
        assert np.array_equal(masked[top:bottom, left:right], rgb[top:bottom, left:right]), (case_id, "误遮区域必须保留原像素")


def test_single_class_confirmation_matches_the_same_real_models_running_both_classes():
    """按类复核须与原双类推理中的同类框、得分完全一致。"""
    manifest = json.loads(Path("tests/fixtures/quality-cases.json").read_text("utf-8"))
    case = next(case for case in manifest["cases"] if case["id"] == "image-7")
    source = Path(manifest["local_source"]) / case["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    with Image.open(source) as image:
        pixels = np.array(image.convert("RGB").crop((2090, 0, 2320, 180)).resize((320, 250)))
    detector = load_detector(Settings(_env_file=None, detection_backend="quality").model_dump())
    for model in (detector.primary, detector.detail):
        both = model.candidates(pixels)
        assert any(d.kind == "face" for d in both), "复核样本必须包含真实人脸候选"
        for kind in ("face", "plate"):
            assert model.candidates(pixels, kind=kind) == [d for d in both if d.kind == kind]
