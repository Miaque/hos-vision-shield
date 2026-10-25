"""Conservative original-pixel evidence for extra plate recovery, never a detection score."""

from contextlib import contextmanager
from contextvars import ContextVar
import math

import cv2
import numpy as np


_SOURCE = ContextVar("plate_source_view", default=None)


@contextmanager
def source_view(gray, *, left=0, top=0, x_scale=1, y_scale=1):
    token = _SOURCE.set((gray, left, top, x_scale, y_scale))
    try:
        yield
    finally:
        _SOURCE.reset(token)


def needs_plate_work(pixels, box):
    source = _SOURCE.get()
    if source is not None:
        pixels, left, top, xs, ys = source
        x1, y1, x2, y2 = box
        box = (left + x1 * xs, top + y1 * ys, left + x2 * xs, top + y2 * ys)
    if not isinstance(pixels, np.ndarray):
        return True
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("Expected original uint8 source pixels")
    x1, y1, x2, y2 = box
    if not all(math.isfinite(value) for value in box) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid original plate coordinates")
    height, width = pixels.shape[:2]
    w, h = x2 - x1, y2 - y1
    # Partial/incomplete candidates cannot establish the full plate's readability.
    if x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1 or not 1.6 <= w / h <= 6:
        return True
    patch = pixels[round(y1 + h * .18):round(y2 - h * .18),
                   round(x1 + w * .15):round(x2 - w * .15)]
    if min(patch.shape[:2]) < 3:
        return True
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    patch = patch.astype(np.float32)
    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    lap = cv2.Laplacian(patch, cv2.CV_32F)
    frequency = math.sqrt(float(np.mean(lap * lap)) / (float(np.mean(dx * dx + dy * dy)) + 1e-6))
    horizontal_detail = math.sqrt(float(np.mean(dx * dx))) / (float(patch.std()) + 1e-6)
    # shortcut: conservative source-crop calibration; revalidate against new cameras and plate styles.
    return not (frequency < .4 and (h * frequency < 4 or h * frequency < 18 and horizontal_detail < 1.5))
