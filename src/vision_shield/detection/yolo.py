"""CUDA face proposals with the existing EgoBlur plate detector and quality gates."""

from pathlib import Path

import cv2
import numpy as np
import torch

from vision_shield.detection import Detection
from vision_shield.detection.egoblur import Detector as EgoBlurDetector, _HalfPlate
from vision_shield.detection.timing import measure, timed

MODEL_PATH = Path(__file__).with_name("assets") / "yolov8n_face.onnx"


class Detector(EgoBlurDetector):
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15, *, face_threshold=0.1, device="cuda"):
        if device != "cuda":
            raise ValueError("quality_yolo requires CUDA inference")
        super().__init__(None, plate_path, max_side, box_padding, face_threshold=face_threshold, device=device)
        if torch.cuda.get_device_capability(self.device)[0] >= 8:
            # Fold static weights on CPU; all forward weights and operations still target the selected CUDA device.
            backbone = torch.jit.load(str(plate_path), map_location="cpu").eval().model.model.backbone
            self.models[1] = _HalfPlate(self.models[1], backbone)
        self.face_path, self.sessions = Path(face_path), {}
        self.providers = [self._session(160, 160).get_providers()[0]]

    @timed("session-primary-face")
    def _session(self, height, width):
        import onnxruntime as ort

        key = (height, width)
        if key not in self.sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            for name, value in (("batch", 1), ("height", height), ("width", width)):
                options.add_free_dimension_override_by_name(name, value)
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(str(self.face_path), sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": self.device.index, "use_tf32": 0,
                                                       "cudnn_conv_algo_search": "HEURISTIC"})])
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Face CUDA provider was not activated")
            # Fixed dimensions fold shape operations; bound sessions for changing source aspect ratios.
            if len(self.sessions) == 4:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[key] = session
        # Keep recurring full-frame sessions hot while local confirmation shapes change.
        self.sessions[key] = self.sessions.pop(key)
        return self.sessions[key]

    def proposals(self, rgb):
        return self.candidates(rgb, kind="face", full_frame=True)

    def candidates(self, rgb, *, kind=None, full_frame=False):
        if isinstance(rgb, torch.Tensor):
            rgb = self.prepare(rgb)
        elif rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("Expected nonempty RGB uint8 pixels")
        detections = []
        if kind != "plate":
            height, width = rgb.shape[:2]
            # Larger enlarged views miss the reviewed side profile; keep local confirmation at 160px.
            side = 160 if kind == "face" and not full_frame else max(32, ((min(self.max_side, max(height, width)) + 31) // 32) * 32)
            scale = side / max(height, width)
            rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
            dx, dy = (side - rw) % 32, (side - rh) % 32
            left, top = round(dx / 2 - 0.1), round(dy / 2 - 0.1)
            if isinstance(rgb, torch.Tensor):
                from vision_shield.detection.cuda_pixels import resize, pad, blob, run

                with torch.cuda.stream(self.stream):
                    rgb.record_stream(self.stream)
                    with measure(self, "prepare-primary"):
                        view = pad(resize(rgb, (rw, rh), "linear"), left, round(dx / 2 + 0.1), top, round(dy / 2 + 0.1), 114)
                        session = self._session(*view.shape[:2])
                        tensor = blob(view, 1 / 255)
                    output = run(session, tensor, self.stream, timing=self)[0]
            else:
                with measure(self, "prepare-primary"):
                    view = cv2.copyMakeBorder(cv2.resize(rgb, (rw, rh)), top, round(dy / 2 + 0.1),
                                             left, round(dx / 2 + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
                    session = self._session(*view.shape[:2])
                    tensor = cv2.dnn.blobFromImage(view, 1 / 255)
                with measure(self, "forward-primary-face"):
                    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            with measure(self, "post-primary"):
                if (output.ndim != 3 or output.shape[:2] != (1, 20) or not np.isfinite(output).all()
                        or (output[:, 4] < 0).any() or (output[:, 4] > 1).any()):
                    raise ValueError("Invalid face detector output")
                # The remaining channels are landmarks; only box coordinates and confidence are consumed.
                rows = output[0, :5].T
                rows = rows[rows[:, 4] > self.face_threshold]
                boxes = rows[:, :4].copy()
                if (boxes[:, 2:] <= 0).any():
                    raise ValueError("Invalid face detector output")
                boxes[:, :2] -= boxes[:, 2:] / 2
                kept = cv2.dnn.NMSBoxes(boxes.tolist(), rows[:, 4].tolist(), self.face_threshold, 0.3)
                for index in np.asarray(kept, dtype=int).reshape(-1)[:1000]:
                    x, y, w, h = boxes[index]
                    box = (max(0, (float(x) - left) / scale), max(0, (float(y) - top) / scale),
                           min(width, (float(x + w) - left) / scale), min(height, (float(y + h) - top) / scale))
                    if box[2] > box[0] and box[3] > box[1]:
                        detections.append(Detection("face", box, float(rows[index, 4])))
        if kind != "face":
            detections.extend(super().candidates(rgb, kind="plate"))
        return detections
