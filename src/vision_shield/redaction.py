"""Shared solid masking in RGB pixel coordinates, independent of media and HTTP."""

import math
from collections.abc import Iterable
from collections import deque

import cv2
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
    """Follow confirmed masks briefly; only actual detections may renew their lifetime."""

    def __init__(self, detector):
        self.detector = detector
        self.history = deque(maxlen=128)
        self.previous, self.shape, self.moment = None, None, None
        self.gray = None

    def apply(self, pixels: np.ndarray, moment: float) -> np.ndarray:
        height, width = pixels.shape[:2]
        rows, cols = np.linspace(0, height - 1, 36, dtype=int), np.linspace(0, width - 1, 64, dtype=int)
        thumbnail = pixels[rows[:, None], cols[None, :]].astype(np.float32)
        reset = (self.shape != pixels.shape or self.moment is None or moment <= self.moment
                 or moment - self.moment > 0.2
                 or np.abs(thumbnail - self.previous).mean() > 20)
        if reset:
            self.history.clear()
            self.gray = None
        self.previous, self.shape, self.moment = thumbnail, pixels.shape, moment
        boxes = self.detector.detect(pixels)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        retained, history = [], []
        for seen, checked, box in self.history:
            if moment - seen > 0.5 or any(self._overlap(box, current) >= 0.2 for current in boxes):
                continue
            predicted = self._follow(gray, box)
            if predicted is None:
                confirmed = self._recheck(pixels, box)
                if confirmed is not None:
                    retained.append(confirmed)
                    history.append((moment, moment, confirmed))
                    continue
                if moment - seen <= 0.2:
                    retained.append(box)
                    history.append((seen, checked, box))
                continue
            if any(self._overlap(predicted, current) >= 0.2 for current in boxes):
                continue
            if moment - seen > 0.2 and moment - checked >= 0.2:
                confirmed = self._recheck(pixels, predicted)
                checked = moment
                if confirmed is not None:
                    predicted, seen = confirmed, moment
            retained.append(predicted)
            history.append((seen, checked, predicted))
        self.history = deque(history + [(moment, moment, box) for box in boxes], maxlen=128)
        self.gray = gray
        return redact(pixels, boxes + retained)

    @staticmethod
    def _overlap(a, b):
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union if union > 0 else 0

    def _follow(self, gray, box):
        if self.gray is None:
            return None
        height, width = gray.shape
        x1, y1, x2, y2 = box
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        patch = self.gray[top:bottom, left:right]
        if min(patch.shape) < 3 or patch.std() < 2:
            return None
        margin = min(96, max(8, max(patch.shape) // 2))
        # shortcut: template matching follows translations; large rotation/scale needs a new detection.
        sx, sy = max(0, left - margin), max(0, top - margin)
        search = gray[sy:min(height, bottom + margin), sx:min(width, right + margin)]
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
        if score < 0.8:
            return None
        dx, dy = sx + location[0] - left, sy + location[1] - top
        return max(0, x1 + dx), max(0, y1 + dy), min(width, x2 + dx), min(height, y2 + dy)

    def _recheck(self, pixels, box):
        x1, y1, x2, y2 = box
        size = max(96, 3 * max(x2 - x1, y2 - y1))
        left, top = max(0, math.floor((x1 + x2 - size) / 2)), max(0, math.floor((y1 + y2 - size) / 2))
        right, bottom = min(pixels.shape[1], math.ceil((x1 + x2 + size) / 2)), min(pixels.shape[0], math.ceil((y1 + y2 + size) / 2))
        crop = pixels[top:bottom, left:right]
        scale = 320 / max(crop.shape[:2])
        view = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        xs, ys = (right - left) / view.shape[1], (bottom - top) / view.shape[0]
        candidates = [(left + a * xs, top + b * ys, left + c * xs, top + d * ys)
                      for a, b, c, d in self.detector.detect(view)]
        candidates = [candidate for candidate in candidates if self._overlap(box, candidate) >= 0.2]
        return max(candidates, key=lambda candidate: self._overlap(box, candidate)) if candidates else None
