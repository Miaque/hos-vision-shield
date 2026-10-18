"""A bounded sample review, never a claim of dataset-wide recall."""

import hashlib
import json
from pathlib import Path

import numpy as np
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
