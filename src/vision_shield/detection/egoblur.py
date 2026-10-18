"""EgoBlur Gen1 dual detector; intentionally provides no recognition output."""

import numpy as np
import torch
import torchvision
from PIL import Image

from vision_shield.detection import Detection


class Detector:
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15, *, face_threshold=0.1):
        torch.set_num_threads(2)
        self.max_side, self.box_padding = max_side, box_padding
        self.face_threshold = face_threshold
        self.models = [torch.jit.load(str(path), map_location="cpu").eval() for path in (face_path, plate_path)]

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Return padded xyxy boxes in source RGB coordinates without altering the pixels."""
        # Mobile-camera blur can extend beyond the predicted rectangle.
        return [(x1 - (x2 - x1) * self.box_padding, y1 - (y2 - y1) * self.box_padding,
                 x2 + (x2 - x1) * self.box_padding, y2 + (y2 - y1) * self.box_padding)
                for detection in self.candidates(rgb) for x1, y1, x2, y2 in [detection.box]]

    def candidates(self, rgb: np.ndarray) -> list[Detection]:
        """Return unpadded source boxes with type and score for cross-model confirmation."""
        height, width = rgb.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        detection_pixels = rgb
        if scale < 1:
            detection_pixels = np.asarray(Image.fromarray(rgb).resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS))
        # The released Gen1 model expects BGR uint8 CHW, without normalization.
        image = torch.from_numpy(detection_pixels[:, :, ::-1].transpose(2, 0, 1).copy())
        x_scale, y_scale = width / detection_pixels.shape[1], height / detection_pixels.shape[0]
        boxes = []
        # Gen1 JIT profiling adds seconds to cold frames without improving this CPU's steady rate.
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            for model, kind, threshold in zip(self.models, ("face", "plate"), (self.face_threshold, 0.1)):
                coordinates, _, scores, _ = model(image)
                if (coordinates.ndim != 2 or coordinates.shape[1] != 4 or scores.shape != (len(coordinates),)
                        or not torch.isfinite(coordinates).all() or not torch.isfinite(scores).all()
                        or (scores < 0).any() or (scores > 1).any()
                        or (coordinates[:, 2:] <= coordinates[:, :2]).any()):
                    raise ValueError("Invalid detector output")
                keep = torchvision.ops.nms(coordinates, scores, 0.3)
                # Map each axis using the actual rounded input size, then mask full-resolution pixels.
                selected = keep[scores[keep] > threshold]
                boxes.extend(Detection(kind, (x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale), score)
                             for (x1, y1, x2, y2), score in zip(coordinates[selected].cpu().tolist(),
                                                               scores[selected].cpu().tolist(), strict=True))
        return boxes
