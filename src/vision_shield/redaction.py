"""Shared opaque masking in RGB pixel coordinates, independent of media and HTTP."""

import math
import time
from collections.abc import Iterable
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

from vision_shield.detection import Detection
from vision_shield.detection.timing import measure, timed


def redact(pixels: np.ndarray, boxes: Iterable[tuple[float, float, float, float]], *, mode: str = "solid") -> np.ndarray:
    """Copy RGB pixels and apply dark gray or coarse mosaic to clipped regions."""
    if mode not in {"solid", "mosaic"}:
        raise ValueError("Unsupported redaction mode")
    result = pixels.copy()
    height, width = result.shape[:2]
    for x1, y1, x2, y2 in boxes:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Non-finite detection coordinates")
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if right > left and bottom > top:
            region = result[top:bottom, left:right]
            if mode == "solid" or min(region.shape[:2]) < 2:
                region[:] = 64
            else:
                # Use original pixels to avoid repeatedly degrading overlapping regions.
                small = cv2.resize(pixels[top:bottom, left:right],
                                   (max(1, min(8, (right - left) // 12)),
                                    max(1, min(8, (bottom - top) // 12))), interpolation=cv2.INTER_AREA)
                region[:] = cv2.resize(small, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    return result


class _Track(NamedTuple):
    seen: float
    checked: float
    box: tuple[float, float, float, float]
    kind: str
    reference: np.ndarray | None = None


class FrameRedactor:
    """Follow confirmed masks briefly; only actual detections may renew their lifetime."""

    def __init__(self, detector, *, mode="solid", detection_interval=1):
        if mode not in {"solid", "mosaic"}:
            raise ValueError("Unsupported redaction mode")
        if type(detection_interval) is not int or detection_interval not in (1, 3):
            raise ValueError("Unsupported detection interval")
        self.mode = mode
        self.detection_interval, self.frame_index = detection_interval, 0
        self.detection_stats = dict(full_frames=0, skipped_frames=0, local_rechecks=0)
        self.detector = detector
        self.stage_times = getattr(detector, "stage_times", None)
        self.history = deque(maxlen=128)
        self.previous, self.shape, self.moment = None, None, None
        self.gray = None
        self.inference_seconds = 0.0
        self.candidate_timings = {}

    def apply(self, pixels: np.ndarray, moment: float) -> np.ndarray:
        return redact(pixels, self.regions(pixels, moment), mode=self.mode)

    def regions(self, pixels, moment, *, thumbnail=None, gray=None):
        height, width = pixels.shape[:2]
        if thumbnail is None:
            rows, cols = np.linspace(0, height - 1, 36, dtype=int), np.linspace(0, width - 1, 64, dtype=int)
            thumbnail = pixels[rows[:, None], cols[None, :]].astype(np.float32)
        reset = (self.shape != pixels.shape or self.moment is None or moment <= self.moment
                 or moment - self.moment > 0.2
                 or np.abs(thumbnail - self.previous).mean() > 20)
        if reset:
            self.history.clear()
            self.gray = None
        elapsed = moment - self.moment if not reset else 0
        self.previous, self.shape, self.moment = thumbnail, pixels.shape, moment
        # shortcut: new entries can wait two frames; keep interval 1 unless offline quality review passes.
        detect = self.frame_index % self.detection_interval == 0
        self.frame_index += 1
        detections = self._merge_detections(self._detections(pixels)) if detect else []
        self.detection_stats["full_frames" if detect else "skipped_frames"] += 1
        boxes = [detection.box for detection in detections]
        if gray is None:
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        retained, history = [], []
        pairs = sorted(((self._overlap(track.box, detection.box), old, new)
                        for old, track in enumerate(self.history) for new, detection in enumerate(detections)
                        if track.kind == detection.kind and moment - track.seen <= 0.5), reverse=True)
        matched_old, matched_new = set(), set()
        for overlap, old, new in pairs:
            if overlap < 0.2:
                break
            if old in matched_old or new in matched_new:
                continue
            matched_old.add(old)
            matched_new.add(new)
            boxes[new] = self._smooth(self.history[old].box, boxes[new], elapsed)
        for old, (seen, checked, box, kind, reference) in enumerate(self.history):
            if moment - seen > 0.5 or old in matched_old:
                continue
            predicted = self._follow(gray, box, reference)
            if predicted is None:
                confirmed = self._recheck(pixels, box, kind) if kind else self._recheck(pixels, box)
                if confirmed is not None:
                    retained.append(confirmed)
                    history.append(_Track(moment, moment, confirmed, kind))
                    continue
                if moment - seen <= 0.2:
                    retained.append(box)
                    # A failed match must keep the last target template, not learn the background.
                    history.append(_Track(seen, checked, box, kind, self.gray if reference is None else reference))
                continue
            matches = [(self._overlap(predicted, detection.box), new)
                       for new, detection in enumerate(detections)
                       if new not in matched_new and detection.kind == kind]
            overlap, new = max(matches, default=(0, -1))
            if overlap >= 0.2:
                matched_new.add(new)
                boxes[new] = self._smooth(predicted, boxes[new], elapsed)
                continue
            if moment - seen > 0.2 and moment - checked >= 0.2:
                confirmed = self._recheck(pixels, predicted, kind) if kind else self._recheck(pixels, predicted)
                checked = moment
                if confirmed is not None:
                    predicted, seen = confirmed, moment
            retained.append(predicted)
            history.append(_Track(seen, checked, predicted, kind))
        self.history = deque(history + [_Track(moment, moment, box, detection.kind)
                                       for box, detection in zip(boxes, detections, strict=True)], maxlen=128)
        self.gray = gray
        return boxes + retained

    def _detections(self, pixels, kind=None):
        started = time.monotonic()
        regions = getattr(self.detector, "detect_regions", None)
        if regions:
            detections = regions(pixels, kind=kind) if kind else regions(pixels)
        else:
            detections = [Detection("", box, 1) for box in self.detector.detect(pixels)]
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            if (regions and detection.kind not in {"face", "plate"}
                    or not all(math.isfinite(v) for v in (*detection.box, detection.score))
                    or not 0 <= detection.score <= 1 or x2 <= x1 or y2 <= y1):
                raise ValueError("Invalid detector output")
        self.inference_seconds += time.monotonic() - started
        for name, seconds in getattr(self.detector, "candidate_timings", {}).items():
            self.candidate_timings[name] = self.candidate_timings.get(name, 0.0) + seconds
        return detections

    @staticmethod
    def _merge_detections(detections):
        # Multiple models/views can describe one target; retain the union, not NMS's smaller mask.
        merged = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            for index, other in enumerate(merged):
                if detection.kind == other.kind and FrameRedactor._overlap(detection.box, other.box) >= 0.6:
                    a, b = detection.box, other.box
                    merged[index] = Detection(other.kind, (min(a[0], b[0]), min(a[1], b[1]),
                                                           max(a[2], b[2]), max(a[3], b[3])), other.score)
                    break
            else:
                merged.append(detection)
        return merged

    @staticmethod
    def _smooth(previous, current, elapsed):
        # Never shrink inside today's detection; bound the trailing margin on moving targets.
        weight = 1 - math.exp(-elapsed / 0.1)
        margin = ((current[2] - current[0]) * 0.25, (current[3] - current[1]) * 0.25)
        blended = [old + weight * (new - old) for old, new in zip(previous, current, strict=True)]
        return (max(current[0] - margin[0], min(current[0], blended[0])),
                max(current[1] - margin[1], min(current[1], blended[1])),
                min(current[2] + margin[0], max(current[2], blended[2])),
                min(current[3] + margin[1], max(current[3], blended[3])))

    @staticmethod
    def _overlap(a, b):
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union if union > 0 else 0

    @timed("track-update")
    def _follow(self, gray, box, reference=None):
        previous = self.gray if reference is None else reference
        if previous is None:
            return None
        height, width = gray.shape
        x1, y1, x2, y2 = box
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        # Mask padding contains stationary background that must not anchor a moving face.
        inset_x, inset_y = (right - left) // 5, (bottom - top) // 5
        left, right = left + inset_x, right - inset_x
        top, bottom = top + inset_y, bottom - inset_y
        patch = previous[top:bottom, left:right]
        if min(patch.shape) < 3 or patch.std() < 2:
            return None
        # Small faces can move farther than their own width between adjacent frames.
        margin = 96
        # shortcut: template matching follows translations; large rotation/scale needs a new detection.
        sx, sy = max(0, left - margin), max(0, top - margin)
        search = gray[sy:min(height, bottom + margin), sx:min(width, right + margin)]
        _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
        if score < 0.8:
            return None
        matched = search[location[1]:location[1] + patch.shape[0],
                         location[0]:location[0] + patch.shape[1]]
        previous_search = previous[sy:sy + search.shape[0], sx:sx + search.shape[1]]
        _, reverse_score, _, reverse = cv2.minMaxLoc(
            cv2.matchTemplate(previous_search, matched, cv2.TM_CCOEFF_NORMED))
        # Repeated background textures must return to the same source patch.
        if reverse_score < 0.8 or abs(sx + reverse[0] - left) > 2 or abs(sy + reverse[1] - top) > 2:
            return None
        dx, dy = sx + location[0] - left, sy + location[1] - top
        return max(0, x1 + dx), max(0, y1 + dy), min(width, x2 + dx), min(height, y2 + dy)

    @timed("track-confirm")
    def _recheck(self, pixels, box, kind=""):
        self.detection_stats["local_rechecks"] += 1
        x1, y1, x2, y2 = box
        size = max(96, 3 * max(x2 - x1, y2 - y1))
        left, top = max(0, math.floor((x1 + x2 - size) / 2)), max(0, math.floor((y1 + y2 - size) / 2))
        right, bottom = min(pixels.shape[1], math.ceil((x1 + x2 + size) / 2)), min(pixels.shape[0], math.ceil((y1 + y2 + size) / 2))
        crop = pixels[top:bottom, left:right]
        if getattr(crop, "is_cuda", False):
            with measure(self, "output-copy"):
                crop = crop.cpu().numpy()
        scale = 320 / max(crop.shape[:2])
        view = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        xs, ys = (right - left) / view.shape[1], (bottom - top) / view.shape[0]
        candidates = [(left + a * xs, top + b * ys, left + c * xs, top + d * ys)
                      for detection in self._detections(view, kind) if detection.kind == kind
                      for a, b, c, d in [detection.box]]
        candidates = [candidate for candidate in candidates if self._overlap(box, candidate) >= 0.2]
        return max(candidates, key=lambda candidate: self._overlap(box, candidate)) if candidates else None
