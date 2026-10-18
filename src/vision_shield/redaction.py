"""Shared solid masking in RGB pixel coordinates, independent of media and HTTP."""

import math
from collections.abc import Iterable
from collections import deque

import numpy as np


def redact(pixels: np.ndarray, boxes: Iterable[tuple[float, float, float, float]]) -> np.ndarray:
    """Copy RGB pixels and fill each clipped xyxy region with opaque black."""
    result = pixels.copy()
    height, width = result.shape[:2]
    for x1, y1, x2, y2 in boxes:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Non-finite detection coordinates")
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if right > left and bottom > top:
            result[top:bottom, left:right] = 0
    return result


class FrameRedactor:
    """Hold at most two actual detections for 200ms of source time per media session."""

    def __init__(self, detector):
        self.detector = detector
        self.history = deque(maxlen=2)
        self.previous, self.shape, self.moment = None, None, None

    def apply(self, pixels: np.ndarray, moment: float) -> np.ndarray:
        height, width = pixels.shape[:2]
        rows, cols = np.linspace(0, height - 1, 36, dtype=int), np.linspace(0, width - 1, 64, dtype=int)
        thumbnail = pixels[rows[:, None], cols[None, :]].astype(np.float32)
        reset = (self.shape != pixels.shape or self.moment is None or moment <= self.moment
                 or moment - self.moment > 0.2
                 or np.abs(thumbnail - self.previous).mean() > 20)
        if reset:
            self.history.clear()
        self.previous, self.shape, self.moment = thumbnail, pixels.shape, moment
        while self.history and moment - self.history[0][0] > 0.2:
            self.history.popleft()
        boxes = self.detector.detect(pixels)
        retained = [box for _, previous_boxes in self.history for box in previous_boxes]
        if boxes:
            self.history.append((moment, boxes))
        # shortcut: short holds cover brief misses, not fast motion or never-detected targets.
        return redact(pixels, boxes + retained)
