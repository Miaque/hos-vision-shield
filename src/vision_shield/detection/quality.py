"""Cross-confirm uncertain detections; inference errors never become partial success."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

import cv2
import numpy as np

from vision_shield.detection.cpu import Detector as DetailDetector
from vision_shield.detection.egoblur import Detector as PrimaryDetector
from vision_shield.detection import Detection
from vision_shield.detection.plate_readability import needs_plate_work
from vision_shield.detection.timing import StageTimes, measure, timed


def _timed_candidates(model, pixels, *, kind=None, full_frame=False):
    started = time.monotonic() if getattr(model, 'stage_times', None) is not None else None
    faces = getattr(model, "proposals", None) if full_frame and kind == "face" else None
    candidates = faces(pixels) if faces else model.candidates(pixels, kind=kind)
    return candidates, time.monotonic() - started if started is not None else 0.0


def _submit_primary(owner, pixels, *, route, **kwargs):
    observer = getattr(getattr(owner, 'stage_times', None), 'primary_observer', None)
    if observer:
        return observer.submit_primary(owner._pool, _timed_candidates, owner.primary, pixels, route=route, **kwargs)
    return owner._pool.submit(_timed_candidates, owner.primary, pixels, **kwargs)


def _primary_result(owner, pending):
    observer = getattr(getattr(owner, 'stage_times', None), 'primary_observer', None)
    return observer.primary_result(pending) if observer else pending.result()


class Detector:
    @timed("plate-work-check")
    def plate_needs_work(self, rgb, box):
        return needs_plate_work(rgb, box)

    @cached_property
    def _pool(self):
        return ThreadPoolExecutor(max_workers=1)

    def __init__(self, config: dict):
        self.max_side, self.box_padding = config["detection_max_side"], config["detection_box_padding"]
        self.inference_device = config.get("inference_device", "cpu")
        primary_type, face_path = PrimaryDetector, config["egoblur_face_model_path"]
        if config["detection_backend"] == "quality_yolo":
            from vision_shield.detection.yolo import Detector as YOLODetector, MODEL_PATH
            primary_type, face_path = YOLODetector, MODEL_PATH
        self.primary = primary_type(face_path, config["egoblur_plate_model_path"], self.max_side, self.box_padding,
                                    face_threshold=0.1, device=self.inference_device)
        # Lower scores may corroborate a candidate, but never start a mask by themselves.
        detail_type = DetailDetector
        if self.inference_device == "cuda":
            from vision_shield.detection.cuda_detail import Detector as CUDADetailDetector
            detail_type = CUDADetailDetector
        # Dense 640px tiles recover tiny faces at high cost; retain larger tiles for 4K coverage.
        self.detail = detail_type(config["face_model_path"], config["plate_model_path"],
                                     self.max_side, self.box_padding, face_threshold=0.3,
                                     parallel=self.inference_device == "cuda", tile_side=1920)
        if config.get('debug', False):
            self.stage_times = self.primary.stage_times = self.detail.stage_times = StageTimes()

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        return [detection.box for detection in self.detect_regions(rgb)]

    def plate_proposals(self, rgb):
        self._proposal_candidates = None
        prepare = getattr(self.primary, "prepare", None)
        pixels = prepare(rgb) if prepare else rgb
        # Proposals trigger the unchanged quality gate; they never create or renew masks.
        if getattr(self, "inference_device", "cpu") != "cuda":
            return self.detail.candidates(pixels, kind="plate")
        # Both outcomes need full faces; share detail views and overlap the independent models.
        pending = _submit_primary(self, pixels, route='plate_proposals', kind="face", full_frame=True)
        try:
            detail, detail_seconds = _timed_candidates(self.detail, pixels)
        finally:
            with measure(self, 'wait-primary'):
                primary, primary_seconds = _primary_result(self, pending)
        self._proposal_candidates = (rgb, primary, primary_seconds, detail, detail_seconds)
        return [detection for detection in detail if detection.kind == "plate"]

    def detect_faces(self, rgb):
        return self.detect_regions(rgb, kind="face", full_frame=True)

    def detect_regions(self, rgb: np.ndarray, *, kind: str | None = None, full_frame=False) -> list[Detection]:
        """Retain strong primary detections and confirmed complementary masks."""
        cached = getattr(self, "_proposal_candidates", None)
        self._proposal_candidates = None
        if cached is not None and cached[0] is rgb and (kind is None or kind == "face" and full_frame):
            _, primary, primary_seconds, detail, detail_seconds = cached
            if kind is None:
                plates, seconds = _timed_candidates(self.primary, self.primary.prepare(rgb), kind="plate")
                primary, primary_seconds = primary + plates, primary_seconds + seconds
            else:
                detail = [detection for detection in detail if detection.kind == "face"]
        elif self.inference_device == "cuda":
            prepare = getattr(self.primary, "prepare", None)
            pixels = prepare(rgb) if prepare else rgb
            # 两路全图检测互不依赖；等待两路结束后再复核，异常不返回部分结果。
            pending = _submit_primary(self, pixels, route='detect_regions', kind=kind, full_frame=full_frame)
            try:
                detail, detail_seconds = _timed_candidates(self.detail, pixels, kind=kind)
            finally:
                with measure(self, 'wait-primary'):
                    primary, primary_seconds = _primary_result(self, pending)
        else:
            primary, primary_seconds = _timed_candidates(self.primary, rgb, kind=kind, full_frame=full_frame)
            detail, detail_seconds = _timed_candidates(self.detail, rgb, kind=kind)
        height, width = rgb.shape[:2]
        boxes = []
        for candidates, others, verifier, is_primary in [(primary, detail, self.detail, True),
                                                        (detail, primary, self.primary, False)]:
            for detection in candidates:
                recovered = False
                if detection.kind == "face" and detection.score < (0.3 if is_primary else 0.6):
                    x1, y1, x2, y2 = detection.box
                    # 用户允许极小目标漏遮；不再为低分微小人脸启动放大复核。
                    if min(x2 - x1, y2 - y1) < 20:
                        continue
                    if is_primary:
                        recovered = (x2 - x1 < 0.75 * (y2 - y1) and min(x2 - x1, y2 - y1) < 128
                                     and not any(other.kind == "face" and other.score >= 0.3
                                                 and self._overlap(detection.box, other.box) >= 0.2 for other in others)
                                     and self._recover_face(rgb, detection))
                    else:
                        # 弱小脸只有在放大视图获得可信主模型确认后才能起框。
                        recovered = (min(x2 - x1, y2 - y1) < 128
                                     and self._confirmed(rgb, detection, self.primary, 0.7, corroborate_mirror=True))
                    if not recovered:
                        continue
                global_score = local_score = 0
                if detection.kind == "plate" and detection.score < 0.7:
                    # 两个低分全图车牌框可能同错，至少一路强分或局部复核才能接受。
                    global_score = 0.7
                if detection.kind == "face":
                    global_score = 0.3
                    x1, y1, x2, y2 = detection.box
                    # 大候选的两路中等分数易被轮毂相互佐证；小侧脸保留原有局部灵敏度。
                    if min(x2 - x1, y2 - y1) >= 128:
                        global_score = local_score = 0.9 if is_primary else 0.7
                        if not is_primary and detection.score >= 0.9:
                            global_score, local_score = 0.3, 0.1
                if not (recovered or is_primary and detection.score >= 0.7
                        or any(other.kind == detection.kind and other.score >= global_score
                               and self._overlap(detection.box, other.box) >= 0.2
                               for other in others)
                        or (detection.kind != "plate" or self.plate_needs_work(rgb, detection.box))
                        and self._confirmed(rgb, detection, verifier, local_score)):
                    continue
                x1, y1, x2, y2 = detection.box
                pad_x = (x2 - x1) * self.box_padding * (2 if not is_primary and detection.kind == "plate" else 1)
                pad_y = (y2 - y1) * self.box_padding
                boxes.append(Detection(detection.kind, (max(0, x1 - pad_x), max(0, y1 - pad_y),
                                       min(width, x2 + pad_x), min(height, y2 + pad_y)), detection.score))
        if getattr(self, 'stage_times', None) is not None:
            self.candidate_timings = {"candidate-primary": primary_seconds, "candidate-detail": detail_seconds}
        # Skipping a weak plate's brief confirmation opportunity can hide it for many frames.
        if kind is None:
            self.unconfirmed_plate_boxes = tuple(
                candidate.box for candidate in primary if candidate.kind == 'plate' and not any(
                    found.kind == 'plate' and self._overlap(candidate.box, found.box) >= .2 for found in boxes)
            )
            self.has_unconfirmed_plates = bool(self.unconfirmed_plate_boxes)
        return boxes

    @timed("local-confirm")
    def _recover_face(self, rgb, detection):
        """Weak small profiles need matching detections from both enlarged local models."""
        x1, y1, x2, y2 = detection.box
        size = max(96, 4 * max(x2 - x1, y2 - y1))
        left, top = max(0, math.floor((x1 + x2 - size) / 2)), max(0, math.floor((y1 + y2 - size) / 2))
        right, bottom = min(rgb.shape[1], math.ceil((x1 + x2 + size) / 2)), min(rgb.shape[0], math.ceil((y1 + y2 + size) / 2))
        crop = rgb[top:bottom, left:right]
        if getattr(crop, "is_cuda", False):
            with measure(self, "output-copy"):
                crop = crop.cpu().numpy()
        if not crop.size:
            return False

        def view(side):
            scale = side / max(crop.shape[:2])
            return cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))

        def source_box(box, pixels, flipped=False):
            a, b, c, d = box
            if flipped:
                a, c = pixels.shape[1] - c, pixels.shape[1] - a
            xs, ys = (right - left) / pixels.shape[1], (bottom - top) / pixels.shape[0]
            return left + a * xs, top + b * ys, left + c * xs, top + d * ys

        primary_view = view(640)
        primary = [source_box(other.box, primary_view) for other in self.primary.candidates(primary_view, kind="face")
                   if other.kind == "face" and other.score >= 0.3
                   and self._overlap(detection.box, source_box(other.box, primary_view)) >= 0.2]
        if not primary:
            return False
        detail_view = view(320)
        # 单侧遮挡具有方向偏差，镜像只作为局部复核，不单独新增目标。
        for flipped in (False, True):
            pixels = np.ascontiguousarray(detail_view[:, ::-1]) if flipped else detail_view
            for other in self.detail.candidates(pixels, kind="face"):
                box = source_box(other.box, pixels, flipped)
                if (other.kind == "face" and other.score >= 0.3 and self._overlap(detection.box, box) >= 0.2
                        and any(self._overlap(box, candidate) >= 0.2 for candidate in primary)):
                    return True
        return False

    @staticmethod
    def _overlap(a, b):
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union

    @timed("local-confirm")
    def _confirmed(self, rgb, detection, verifier, minimum_score=0, *, corroborate_mirror=False):
        x1, y1, x2, y2 = detection.box
        size = max(96, 3 * max(x2 - x1, y2 - y1))
        left, top = max(0, math.floor((x1 + x2 - size) / 2)), max(0, math.floor((y1 + y2 - size) / 2))
        right = min(rgb.shape[1], math.ceil((x1 + x2 + size) / 2))
        bottom = min(rgb.shape[0], math.ceil((y1 + y2 + size) / 2))
        if right <= left or bottom <= top:
            return False
        crop = rgb[top:bottom, left:right]
        if getattr(crop, "is_cuda", False):
            with measure(self, "output-copy"):
                crop = crop.cpu().numpy()
        scale = 320 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
        xs, ys = (right - left) / crop.shape[1], (bottom - top) / crop.shape[0]
        expected = ((x1 - left) / xs, (y1 - top) / ys, (x2 - left) / xs, (y2 - top) / ys)
        matches = [other for other in verifier.candidates(crop, kind=detection.kind)
                   if other.kind == detection.kind and self._overlap(expected, other.box) >= 0.2]
        if any(other.score >= minimum_score for other in matches):
            return True
        normal = [other for other in matches if other.score >= 0.3]
        if not corroborate_mirror or not normal:
            return False
        # 无高分时，两种方向的局部结果须同时确认同一位置，单个中等分数不能起框。
        for other in verifier.candidates(np.ascontiguousarray(crop[:, ::-1]), kind=detection.kind):
            a, b, c, d = other.box
            box = (crop.shape[1] - c, b, crop.shape[1] - a, d)
            if (other.kind == detection.kind and other.score >= 0.3 and self._overlap(expected, box) >= 0.2
                    and any(self._overlap(candidate.box, box) >= 0.2 for candidate in normal)):
                return True
        return False
