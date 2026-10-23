"""CUDA inference for the same YuNet/HyperLPR weights and CPU view/postprocessing contract."""

from itertools import batched

import cv2
import numpy as np
import torch

from vision_shield.detection.cpu import Detector as CPUDetector, decode_plates
from vision_shield.detection.timing import measure, timed


def _batch_plate_model(model):
    import onnx

    model.graph.ClearField("value_info")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    for output in model.graph.output:
        output.type.tensor_type.shape.dim[0].dim_param = "batch"
    shapes = {node.input[1] for node in model.graph.node if node.op_type == "Reshape"}
    patched = set()
    # shortcut: fixed HyperLPR export; recheck reshape contracts when changing weights.
    for tensor in model.graph.initializer:
        if tensor.name in shapes:
            shape = onnx.numpy_helper.to_array(tensor).copy()
            if shape.ndim != 1 or not len(shape) or shape[0] != 1:
                raise ValueError("Unsupported plate batch export")
            shape[0] = 0
            tensor.CopyFrom(onnx.numpy_helper.from_array(shape, name=tensor.name))
            patched.add(tensor.name)
    if patched != shapes:
        raise ValueError("Unsupported plate batch export")
    return model.SerializeToString()


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
        self.face_model, self.face_sessions = model.SerializeToString(), {}
        self.plate_model, self.plate_sessions = _batch_plate_model(onnx.load(str(plate_path))), {}
        self.face = ort.InferenceSession(self.face_model, sess_options=options, providers=providers)
        self.plate = ort.InferenceSession(str(plate_path), sess_options=options, providers=providers)
        for session in (self.face, self.plate):
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("YuNet/HyperLPR CUDA provider was not activated")
        self.providers = [self.face.get_providers()[0], self.plate.get_providers()[0]]
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.view_stream, self.face_stream, self.plate_stream = [torch.cuda.Stream(device=self.device) for _ in range(3)]

    def _face_session(self, height, width):
        import onnxruntime as ort

        key = (height, width)
        if key not in self.face_sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.add_free_dimension_override_by_name("height", height)
            options.add_free_dimension_override_by_name("width", width)
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(self.face_model, sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": self.device.index, "use_tf32": 0,
                                                       "cudnn_conv_algo_search": "HEURISTIC"})])
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Face CUDA provider was not activated")
            # Fixed shapes let ORT fold metadata work; changing source sizes cannot grow sessions forever.
            if len(self.face_sessions) == 4:
                self.face_sessions.pop(next(iter(self.face_sessions)))
            self.face_sessions[key] = session
        return self.face_sessions[key]

    def candidates(self, rgb, *, kind=None):
        if isinstance(rgb, torch.Tensor):
            from vision_shield.detection.cuda_pixels import upload

            return self._candidates(upload(rgb, self.device, timing=self), kind=kind)
        return super().candidates(rgb, kind=kind)

    def _plate_session(self, batch):
        import onnxruntime as ort

        if batch == 1:
            return self.plate
        if batch not in self.plate_sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.add_free_dimension_override_by_name("batch", batch)
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(self.plate_model, sess_options=options,
                providers=[("CUDAExecutionProvider", {"device_id": self.device.index, "use_tf32": 0,
                                                       "cudnn_conv_algo_search": "HEURISTIC"})])
            session.disable_fallback()
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("Plate CUDA provider was not activated")
            self.plate_sessions[batch] = session
        return self.plate_sessions[batch]

    def _view_detections(self, rgb, *, kind=None):
        if not isinstance(rgb, torch.Tensor) or not self.parallel or kind == "face" or max(rgb.shape[:2]) <= self.tile_side:
            yield from super()._view_detections(rgb, kind=kind)
            return
        height, width = rgb.shape[:2]
        for bounds in batched(self._views(width, height, self.tile_side), 4):
            prepared = []
            for left, top, right, bottom in bounds:
                source = rgb[top:bottom, left:right]
                prepared.append((left, top, source.shape[:2], self._view(source)))
            if len(prepared) == 1:
                left, top, shape, view = prepared[0]
                yield left, top, shape, view.shape[:2], self._detect_view(view, kind=kind)
                continue
            views = [item[3] for item in prepared]
            pending = self._pool.submit(lambda: [self._detect_view(view, kind="face") for view in views]) if kind != "plate" else None
            try:
                inputs = [self._plate_input(view) for view in views]
                with torch.cuda.stream(self.plate_stream):
                    predictions = self._plate_forward(torch.cat([value[0] for value in inputs]))
                with measure(self, "post-detail"):
                    if predictions.shape != (len(views), 6300, 15):
                        raise ValueError("Invalid plate detector output")
                    plates = [decode_plates(predictions[i:i + 1], view.shape[1], view.shape[0], *inputs[i][1:])
                              for i, view in enumerate(views)]
            finally:
                faces = pending.result() if pending else [[] for _ in views]
            for (left, top, shape, view), face, plate in zip(prepared, faces, plates, strict=True):
                yield left, top, shape, view.shape[:2], face + plate

    @timed("prepare-detail")
    def _view(self, rgb):
        if not isinstance(rgb, torch.Tensor):
            return super()._view(rgb)
        from vision_shield.detection.cuda_pixels import resize

        with torch.cuda.stream(self.view_stream):
            rgb.record_stream(self.view_stream)
            view = rgb.flip(2)
            height, width = view.shape[:2]
            scale = min(1, self.max_side / max(width, height))
            if scale < 1:
                view = resize(view, (max(1, round(width * scale)), max(1, round(height * scale))), "area")
        with measure(self, "explicit-sync"):
            self.view_stream.synchronize()
        return view

    def _faces(self, pixels):
        height, width = pixels.shape[:2]
        if isinstance(pixels, torch.Tensor):
            from vision_shield.detection.cuda_pixels import pad, blob, run

            with torch.cuda.stream(self.face_stream):
                with measure(self, "prepare-detail"):
                    pixels.record_stream(self.face_stream)
                    padded = pad(pixels, 0, (-width) % 32, 0, (-height) % 32)
                    session = self._face_session(*padded.shape[:2])
                    tensor = blob(padded)
                outputs = run(session, tensor, self.face_stream, timing=self, stage="forward-detail-face")
            with measure(self, "post-detail"):
                return decode_faces(dict(zip((o.name for o in session.get_outputs()), outputs)),
                                    padded.shape[1], padded.shape[0], self.face_threshold)
        with measure(self, "prepare-detail"):
            padded = cv2.copyMakeBorder(pixels, 0, (-height) % 32, 0, (-width) % 32, cv2.BORDER_CONSTANT)
            tensor = cv2.dnn.blobFromImage(padded)
        with measure(self, "forward-detail-face"):
            outputs = self.face.run(None, {self.face.get_inputs()[0].name: tensor})
        with measure(self, "post-detail"):
            return decode_faces(dict(zip((o.name for o in self.face.get_outputs()), outputs)),
                                padded.shape[1], padded.shape[0], self.face_threshold)

    def _plate_forward(self, blob):
        if isinstance(blob, torch.Tensor):
            from vision_shield.detection.cuda_pixels import run

            return run(self._plate_session(blob.shape[0]), blob, self.plate_stream, timing=self, stage="forward-detail-plate")[0]
        with measure(self, "forward-detail-plate"):
            return self.plate.run(None, {self.plate.get_inputs()[0].name: blob})[0]

    @timed("prepare-detail")
    def _plate_input(self, bgr):
        if not isinstance(bgr, torch.Tensor):
            return super()._plate_input(bgr)
        from vision_shield.detection.cuda_pixels import resize, pad, blob

        height, width = bgr.shape[:2]
        scale = min(320 / width, 320 / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (320 - resized_width) // 2, (320 - resized_height) // 2
        with torch.cuda.stream(self.plate_stream):
            bgr.record_stream(self.plate_stream)
            view = resize(bgr, (resized_width, resized_height), "linear")
            canvas = pad(view, left, 320 - resized_width - left, top, 320 - resized_height - top)
            tensor = blob(canvas.flip(2), 1 / 255)
        return tensor, resized_width, resized_height, left, top
