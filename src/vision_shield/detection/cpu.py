"""Shared face/plate views and decoding with the default OpenCV CPU inference."""

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from vision_shield.detection import Detection

FACE_THRESHOLD = 0.3
PLATE_THRESHOLD = 0.25


class Detector:
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15, *, face_threshold=FACE_THRESHOLD,
                 parallel=False, tile_side=640):
        cv2.setNumThreads(2)
        self.max_side, self.box_padding = max_side, box_padding
        self.parallel = parallel
        self.tile_side = tile_side
        self.face = cv2.FaceDetectorYN.create(str(face_path), "", (320, 320), face_threshold, 0.3, 5000,
                                            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        self.plate = cv2.dnn.readNetFromONNX(str(plate_path))
        self.plate.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.plate.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        return [detection.box for detection in self.detect_regions(rgb)]

    def detect_regions(self, rgb: np.ndarray, *, kind: str | None = None) -> list[Detection]:
        """Return padded source xyxy masks from a global view and overlapping source tiles."""
        height, width = rgb.shape[:2]
        boxes = []
        for detection in self.candidates(rgb, kind=kind):
            x1, y1, x2, y2 = detection.box
            # Plates need more horizontal margin when blurred or partly occluded.
            pad_x = (x2 - x1) * self.box_padding * (2 if detection.kind == "plate" else 1)
            pad_y = (y2 - y1) * self.box_padding
            boxes.append(Detection(detection.kind, (max(0, x1 - pad_x), max(0, y1 - pad_y),
                                   min(width, x2 + pad_x), min(height, y2 + pad_y)), detection.score))
        # Keep overlapping masks: suppressing a wider, lower-score box can expose a plate edge.
        return boxes

    def candidates(self, rgb: np.ndarray, *, kind: str | None = None) -> list[Detection]:
        """Keep type and score until the quality backend has confirmed uncertain masks."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("Expected nonempty RGB uint8 pixels")
        return self._candidates(rgb, kind=kind)

    def _candidates(self, rgb, *, kind=None):
        height, width = rgb.shape[:2]
        boxes = []
        for left, top, right, bottom in self._views(width, height, self.tile_side):
            source = rgb[top:bottom, left:right]
            view_height, view_width = source.shape[:2]
            view = self._view(source)
            x_scale, y_scale = view_width / view.shape[1], view_height / view.shape[0]
            for detection in self._detect_view(view, kind=kind):
                x1, y1, x2, y2 = detection.box
                if (detection.kind not in {"face", "plate"}
                        or not all(math.isfinite(v) for v in (*detection.box, detection.score))
                        or not 0 <= detection.score <= 1 or x2 <= x1 or y2 <= y1):
                    raise ValueError("Invalid detector output")
                x1, x2 = max(0, x1), min(view.shape[1], x2)
                y1, y2 = max(0, y1), min(view.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                x1, x2 = left + x1 * x_scale, left + x2 * x_scale
                y1, y2 = top + y1 * y_scale, top + y2 * y_scale
                boxes.append(Detection(detection.kind, (x1, y1, x2, y2), detection.score))
        return boxes

    def _view(self, rgb):
        view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = view.shape[:2]
        scale = min(1, self.max_side / max(width, height))
        if scale < 1:
            view = cv2.resize(view, (max(1, round(width * scale)), max(1, round(height * scale))),
                              interpolation=cv2.INTER_AREA)
        return view

    @staticmethod
    def _views(width, height, tile_side=640):
        yield 0, 0, width, height
        if max(width, height) <= tile_side:
            return

        def spans(length):
            # At most 8 positions per axis; enlarge source tiles instead of skipping image regions.
            span = min(length, max(tile_side, math.ceil(length / 6.6)))
            count = min(8, math.ceil((length - span) / (0.8 * span)) + 1)
            return [(int(start), int(start) + span) for start in np.linspace(0, length - span, count).round()]

        for top, bottom in spans(height):
            for left, right in spans(width):
                yield left, top, right, bottom

    def _detect_view(self, bgr: np.ndarray, *, kind: str | None = None) -> list[Detection]:
        if self.parallel and kind is None:
            # 两个独立模型各占一路；退出时等待后台结束，仍按人脸、车牌顺序返回。
            with ThreadPoolExecutor(max_workers=1) as pool:
                faces = pool.submit(self._detect_view, bgr, kind="face")
                plates = self._detect_view(bgr, kind="plate")
                return faces.result() + plates
        height, width = bgr.shape[:2]
        detections = []
        # 局部复核不运行无关类别；全图默认仍执行两类检测。
        if kind != "plate":
            # YuNet requires a minimum spatial extent; padding never invents image details.
            face_input = bgr if getattr(bgr, "is_cuda", False) else cv2.copyMakeBorder(
                bgr, 0, max(0, 32 - height), 0, max(0, 32 - width), cv2.BORDER_CONSTANT)
            faces = self._faces(face_input)
            if faces is not None:
                if faces.ndim != 2 or faces.shape[1] != 15 or not np.isfinite(faces).all():
                    raise ValueError("Invalid face detector output")
                detections.extend(Detection("face", (float(x), float(y), float(x + w), float(y + h)), float(row[-1]))
                                  for row in faces for x, y, w, h in [row[:4]])
        if kind == "face":
            return detections

        # HyperLPR's detection-only 320 model expects letterboxed RGB / 255, not OCR tensors.
        blob, resized_width, resized_height, left, top = self._plate_input(bgr)
        predictions = self._plate_forward(blob)
        if predictions.shape != (1, 6300, 15) or not np.isfinite(predictions).all():
            raise ValueError("Invalid plate detector output")
        rows = predictions[0]
        scores = rows[:, 4] * rows[:, 13:15].max(axis=1)
        selected = np.flatnonzero(scores > PLATE_THRESHOLD)
        boxes = rows[selected, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        if np.any(boxes[:, 2:] <= 0):
            raise ValueError("Invalid plate dimensions")
        for index in np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores[selected].tolist(), PLATE_THRESHOLD, 0.3)).reshape(-1):
            x, y, w, h = boxes[index]
            detections.append(Detection("plate", ((float(x) - left) * width / resized_width,
                                                  (float(y) - top) * height / resized_height,
                                                  (float(x + w) - left) * width / resized_width,
                                                  (float(y + h) - top) * height / resized_height),
                                        float(scores[selected[index]])))
        return detections

    def _plate_input(self, bgr):
        height, width = bgr.shape[:2]
        scale = min(320 / width, 320 / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (320 - resized_width) // 2, (320 - resized_height) // 2
        canvas = np.zeros((320, 320, 3), dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(bgr, (resized_width, resized_height))
        return cv2.dnn.blobFromImage(canvas, 1 / 255, (320, 320), swapRB=True), resized_width, resized_height, left, top

    def _faces(self, pixels):
        self.face.setInputSize((pixels.shape[1], pixels.shape[0]))
        return self.face.detect(pixels)[1]

    def _plate_forward(self, blob):
        self.plate.setInput(blob)
        return self.plate.forward()
