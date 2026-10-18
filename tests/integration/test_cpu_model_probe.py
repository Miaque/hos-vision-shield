"""Real detection-only weights and a real face; never replace missing models with stubs."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vision_shield.config import Settings
from vision_shield.detection import load_detector


def test_cpu_models_match_manifest_and_mask_a_real_face():
    manifest = json.loads(Path("models/cpu-manifest.json").read_text("utf-8"))
    for entry in manifest["models"]:
        with (Path("models") / entry["path"]).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"]
    detector = load_detector(Settings(_env_file=None, detection_backend="cpu").model_dump())
    assert detector.detect(np.zeros((64, 96, 3), np.uint8)) == []
    with Image.open("samples/official-demo.jpg") as source:
        pixels = np.asarray(source.convert("RGB"))
    from vision_shield.redaction import redact
    masked = redact(pixels, detector.detect(pixels))
    assert (masked[270, 560] == 0).all()
