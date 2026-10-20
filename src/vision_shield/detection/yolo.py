"""CPU-only YOLO face/plate inference using the pinned ONNX export contracts."""

import ast

import cv2
import numpy as np
import onnxruntime as ort

from vision_shield.detection.cpu import Detector as DetailDetector


def prepare(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, float, int, int]]:
    """Letterbox RGB without swapping channels; retain actual rounded x/y scales."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Expected nonempty RGB uint8 pixels")
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - rw) // 2, (size - rh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + rh, left:left + rw] = cv2.resize(rgb, (rw, rh))
    return cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=False), (rw / width, rh / height, left, top)


def decode(output: np.ndarray, geometry: tuple, width: int, height: int,
           *, face: bool = False) -> list[tuple[tuple[float, float, float, float], float]]:
    """Decode pinned face post-NMS xyxy or plate raw xywh into view coordinates."""
    expected = output.ndim == 3 and output.shape[0] == 1
    expected = expected and (output.shape[1:] == (300, 21) if face else output.shape[1] == 5)
    if not expected or not np.isfinite(output).all():
        raise ValueError("Invalid YOLO output shape or values")
    rows = output[0] if face else output[0].T
    if np.any((rows[:, 4] < 0) | (rows[:, 4] > 1)):
        raise ValueError("Invalid YOLO score")
    rows = rows[rows[:, 4] >= 0.25]
    if not len(rows):
        return []
    boxes, scores = rows[:, :4].copy(), rows[:, 4]
    if face:
        if np.any(rows[:, 5] != 0):
            raise ValueError("Unexpected YOLO face class")
        boxes[:, 2:] -= boxes[:, :2]
    else:
        boxes[:, :2] -= boxes[:, 2:] / 2
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError("Invalid YOLO box dimensions")
    indices = range(len(boxes)) if face else np.asarray(
        cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.25, 0.45)).reshape(-1)
    xs, ys, left, top = geometry
    detections = []
    for index in indices:
        x, y, w, h = map(float, boxes[index])
        box = (max(0, (x - left) / xs), max(0, (y - top) / ys),
               min(width, (x + w - left) / xs), min(height, (y + h - top) / ys))
        if box[2] > box[0] and box[3] > box[1]:
            detections.append((box, float(scores[index])))
    return detections


class Detector:
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15):
        # Both pinned networks use 640 per view; max_side is accepted for the shared loader only.
        cv2.setNumThreads(2)
        self.max_side, self.box_padding = max_side, box_padding
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.models = []
        for path, name in ((face_path, 'face'), (plate_path, 'License_Plate')):
            model = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            model.disable_fallback()
            metadata = model.get_modelmeta().custom_metadata_map
            if ast.literal_eval(metadata.get('names', '{}')) != {0: name}:
                raise ValueError("Unexpected YOLO model classes")
            if len(model.get_inputs()) != 1 or len(model.get_outputs()) != 1:
                raise ValueError("Unexpected YOLO model interface")
            self.models.append(model)

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Return clipped padded source xyxy boxes; either model's failure aborts the frame."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("Expected nonempty RGB uint8 pixels")
        height, width = rgb.shape[:2]
        result = []
        for left, top, right, bottom in DetailDetector._views(width, height):
            view = rgb[top:bottom, left:right]
            blob, geometry = prepare(view, 640)
            for index, model in enumerate(self.models):
                output = model.run(None, {model.get_inputs()[0].name: blob})[0]
                for box, _ in decode(output, geometry, right-left, bottom-top, face=index == 0):
                    x1, y1, x2, y2 = box
                    px = (x2-x1) * self.box_padding * (2 if index else 1)
                    py = (y2-y1) * self.box_padding
                    result.append((max(0, left+x1-px), max(0, top+y1-py),
                                   min(width, left+x2+px), min(height, top+y2+py)))
        return result
