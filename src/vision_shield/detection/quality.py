"""Cross-confirm uncertain detections; inference errors never become partial success."""

import math

import cv2
import numpy as np

from vision_shield.detection.cpu import Detector as DetailDetector
from vision_shield.detection.egoblur import Detector as PrimaryDetector


class Detector:
    def __init__(self, config: dict):
        self.max_side, self.box_padding = config["detection_max_side"], config["detection_box_padding"]
        self.primary = PrimaryDetector(config["egoblur_face_model_path"], config["egoblur_plate_model_path"],
                                       self.max_side, self.box_padding, face_threshold=0.1)
        # Lower scores may corroborate a candidate, but never start a mask by themselves.
        self.detail = DetailDetector(config["face_model_path"], config["plate_model_path"],
                                     self.max_side, self.box_padding, face_threshold=0.3)

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Retain strong primary detections and confirmed complementary masks."""
        primary, detail = self.primary.candidates(rgb), self.detail.candidates(rgb)
        height, width = rgb.shape[:2]
        boxes = []
        for candidates, others, verifier, is_primary in [(primary, detail, self.detail, True),
                                                        (detail, primary, self.primary, False)]:
            for detection in candidates:
                if detection.kind == "face" and detection.score < (0.3 if is_primary else 0.6):
                    continue
                if not (is_primary and detection.score >= 0.7
                        or any(other.kind == detection.kind and self._overlap(detection.box, other.box) >= 0.2
                               for other in others)
                        or self._confirmed(rgb, detection, verifier)):
                    continue
                x1, y1, x2, y2 = detection.box
                pad_x = (x2 - x1) * self.box_padding * (2 if not is_primary and detection.kind == "plate" else 1)
                pad_y = (y2 - y1) * self.box_padding
                boxes.append((max(0, x1 - pad_x), max(0, y1 - pad_y),
                              min(width, x2 + pad_x), min(height, y2 + pad_y)))
        return boxes

    @staticmethod
    def _overlap(a, b):
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union

    def _confirmed(self, rgb, detection, verifier):
        x1, y1, x2, y2 = detection.box
        size = max(96, 3 * max(x2 - x1, y2 - y1))
        left, top = max(0, math.floor((x1 + x2 - size) / 2)), max(0, math.floor((y1 + y2 - size) / 2))
        right = min(rgb.shape[1], math.ceil((x1 + x2 + size) / 2))
        bottom = min(rgb.shape[0], math.ceil((y1 + y2 + size) / 2))
        if right <= left or bottom <= top:
            return False
        crop = rgb[top:bottom, left:right]
        scale = 320 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        xs, ys = (right - left) / crop.shape[1], (bottom - top) / crop.shape[0]
        expected = ((x1 - left) / xs, (y1 - top) / ys, (x2 - left) / xs, (y2 - top) / ys)
        return any(other.kind == detection.kind and self._overlap(expected, other.box) >= 0.2
                   for other in verifier.candidates(crop))
