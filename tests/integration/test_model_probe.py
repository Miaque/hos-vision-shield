"""T005: real weights only; missing files are failures, not skipped tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_shield.config import Settings

pytestmark = pytest.mark.integration


def test_both_official_models_load_and_return_valid_boxes():
    import torch
    import torchvision  # Registers the operators used by the TorchScript weights.

    settings = Settings()
    manifest_path = Path("models/manifest.json")
    assert manifest_path.is_file(), "Verified official model manifest not found"
    hashes = {item["path"]: item["sha256"] for item in json.loads(manifest_path.read_text("utf-8"))["models"]}
    torch.set_num_threads(2)
    samples = (Path("samples/official-demo.jpg"), Path("samples/opencv-licenseplate.jpg"))
    for path, sample in zip((settings.face_model_path, settings.plate_model_path), samples, strict=True):
        assert path.is_file(), f"Required Gen1 model not found: {path}"
        assert sample.is_file(), f"Official probe sample not found: {sample}"
        with Image.open(sample) as source:
            source = source.convert("RGB")
            source.thumbnail((640, 640))
            # Gen1 consumes BGR uint8 CHW, matching the official OpenCV example.
            pixels = np.array(source)[:, :, ::-1].copy().transpose(2, 0, 1)
        image = torch.from_numpy(pixels)
        with path.open("rb") as model_file:
            assert hashlib.file_digest(model_file, "sha256").hexdigest() == hashes[path.name]
        model = torch.jit.load(str(path), map_location="cpu").eval()
        with torch.inference_mode():
            boxes, classes, scores, dimensions = model(image)
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert len(boxes) == len(scores) == len(classes)
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
