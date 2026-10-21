"""CUDA inference for the same YuNet/HyperLPR weights and CPU view/postprocessing contract."""

import cv2
import numpy as np

from vision_shield.detection.cpu import Detector as CPUDetector


def decode_faces(outputs, width, height, threshold):
    """Match OpenCV FaceDetectorYN decoding, including integer-box NMS."""
    faces = []
    for stride in (8, 16, 32):
        count = (width // stride) * (height // stride)
        values = [outputs[f"{name}_{stride}"] for name in ("cls", "obj", "bbox", "kps")]
        for value, size in zip(values, (1, 1, 4, 10)):
            if value.shape != (1, count, size) or not np.isfinite(value).all():
                raise ValueError("Invalid face detector output")
        cls, obj, boxes, points = (value[0] for value in values)
        scores = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
        selected = np.flatnonzero(scores >= threshold)
        rows = np.empty((len(selected), 15), np.float32)
        grid = np.column_stack((selected % (width // stride), selected // (width // stride))).astype(np.float32)
        sizes = np.exp(boxes[selected, 2:]) * stride
        rows[:, :2] = (grid + boxes[selected, :2]) * stride - sizes / 2
        rows[:, 2:4] = sizes
        rows[:, 4:14] = ((points[selected].reshape(-1, 5, 2) + grid[:, None, :]) * stride).reshape(-1, 10)
        rows[:, 14] = scores[selected]
        faces.append(rows)
    faces = np.concatenate(faces)
    if not np.isfinite(faces).all():
        raise ValueError("Invalid face detector output")
    if len(faces) > 1:
        kept = cv2.dnn.NMSBoxes(np.trunc(faces[:, :4]).astype(int).tolist(),
                               faces[:, 14].tolist(), threshold, 0.3, top_k=5000)
        faces = faces[np.asarray(kept, dtype=int).reshape(-1)]
    return faces if len(faces) else None


class Detector(CPUDetector):
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15, *, face_threshold=0.3,
                 parallel=True, tile_side=640):
        import torch
        import onnx
        import onnxruntime as ort

        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("YuNet/HyperLPR require ONNX Runtime CUDA")
        cv2.setNumThreads(2)
        self.max_side, self.box_padding, self.parallel = max_side, box_padding, parallel
        self.tile_side = tile_side
        self.face_threshold = face_threshold
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                               "use_tf32": 0, "cudnn_conv_algo_search": "HEURISTIC"})]
        # YuNet is convolutional; OpenCV also overrides its fixed input metadata for each view.
        model = onnx.load(str(face_path))
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[2].dim_param, dims[3].dim_param = "height", "width"
        model.graph.ClearField("value_info")
        for output in model.graph.output:
            output.type.tensor_type.shape.dim[1].dim_param = output.name + "_anchors"
        self.face = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=providers)
        self.plate = ort.InferenceSession(str(plate_path), sess_options=options, providers=providers)
        for session in (self.face, self.plate):
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("YuNet/HyperLPR CUDA provider was not activated")
        self.providers = [self.face.get_providers()[0], self.plate.get_providers()[0]]

    def _faces(self, pixels):
        height, width = pixels.shape[:2]
        padded = cv2.copyMakeBorder(pixels, 0, (-height) % 32, 0, (-width) % 32, cv2.BORDER_CONSTANT)
        outputs = self.face.run(None, {self.face.get_inputs()[0].name: cv2.dnn.blobFromImage(padded)})
        return decode_faces(dict(zip((o.name for o in self.face.get_outputs()), outputs)),
                            padded.shape[1], padded.shape[0], self.face_threshold)

    def _plate_forward(self, blob):
        return self.plate.run(None, {self.plate.get_inputs()[0].name: blob})[0]
