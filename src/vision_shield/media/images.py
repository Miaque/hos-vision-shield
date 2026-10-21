import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import TYPE_CHECKING

from vision_shield.redaction import redact
if TYPE_CHECKING:
    from vision_shield.detection import Detector


def process_image(source: str | Path, destination: str | Path, detector: "Detector", max_pixels: int, *, redaction_mode: str = "solid") -> int:
    """Write a fresh JPEG/PNG with normalized orientation; return the number of boxes."""
    with Image.open(source) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Unsupported image format")
        format_name = image.format
        if image.width * image.height > max_pixels:
            raise ValueError("Image pixel limit exceeded")
        pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    boxes = detector.detect(pixels)
    # Create a fresh image so original EXIF thumbnails cannot escape redaction.
    result = Image.fromarray(redact(pixels, boxes, mode=redaction_mode))
    result.save(destination, format=format_name)
    return len(boxes)
