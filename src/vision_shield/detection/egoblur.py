"""EgoBlur Gen1 dual detector; intentionally provides no recognition output."""

from contextlib import nullcontext

import numpy as np
import torch
import torchvision
from PIL import Image

from vision_shield.detection import Detection
from vision_shield.detection.timing import measure


@torch.jit.script
def _single_image_boxes(boxes: torch.Tensor):
    return torch.cat((torch.zeros_like(boxes[:, :1]), boxes), 1)


def _plate_forward(model):
    # shortcut: fixed Gen1/TorchScript internals; revalidate the graph and APIs on model or PyTorch upgrades.
    graph = model.inlined_graph.copy()
    moves = [node for node in graph.findAllNodes("aten::to")
             if node.schema().startswith("aten::to.prim_Device(")]
    if len(moves) != 17:
        raise RuntimeError("Unexpected Gen1 device-transfer graph")
    with graph.insert_point_guard(moves[0]):
        asynchronous = graph.insertConstant(True)
    for node in moves:
        node.replaceInput(3, asynchronous)
    formats = []
    for node in graph.findAllNodes("aten::cat"):
        columns = list(node.inputsAt(0).node().inputs())
        if len(columns) != 2 or columns[0].node().kind() != "aten::unsqueeze":
            continue
        index = columns[0].node().inputsAt(0).node()
        while index.kind() == "aten::slice":
            index = index.inputsAt(0).node()
        if index.kind() == "aten::repeat_interleave":
            formats.append((node, columns[1]))
    if len(formats) != 1 or len(graph.findAllNodes("aten::repeat_interleave")) != 1:
        raise RuntimeError("Unexpected Gen1 single-image pooler graph")
    node, boxes = formats[0]
    # Gen1 consumes one image; each original ROI's batch index is zero, including an empty set.
    with graph.insert_point_guard(node):
        formatted, = graph.insertGraph(_single_image_boxes.graph, [boxes])
    node.output().replaceAllUsesWith(formatted)
    node.destroy()
    torch._C._jit_pass_dce(graph)
    graph.lint()
    return torch._C._create_function_from_graph("plate_inference_forward", graph)


class _HalfPlate(torch.nn.Module):
    """Keep the original graph and float normalization while using half model weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        core = model.model.model
        mean, std = core.pixel_mean.clone(), core.pixel_std.clone()
        model.half()
        core.pixel_mean, core.pixel_std = mean, std
        # Internal transfers target CUDA; their consumers remain ordered on the model stream.
        self._forward = _plate_forward(model)

    def forward(self, pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            boxes, labels, scores, size = self._forward(self.model, pixels)
        return boxes.float(), labels, scores.float(), size


class Detector:
    def __init__(self, face_path, plate_path, max_side=1280, box_padding=0.15, *, face_threshold=0.1, device="cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("推理设备仅支持 cpu 或 cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查显卡驱动及 PyTorch CUDA 构建")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")
        self.stream = torch.cuda.Stream(device=self.device) if device == "cuda" else None
        # 本机交错实测4线程显著快于2线程，避免继续加线程挤占其他媒体进程。
        torch.set_num_threads(4)
        self.max_side, self.box_padding = max_side, box_padding
        self.face_threshold = face_threshold
        # Gen1 的尺寸常量必须留在 CPU；按官方流程仅迁移参数和缓冲区。
        self.models = [torch.jit.load(str(path), map_location="cpu").to(self.device).eval() if path is not None else None
                       for path in (face_path, plate_path)]

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        return [detection.box for detection in self.detect_regions(rgb)]

    def prepare(self, rgb):
        from vision_shield.detection.cuda_pixels import upload

        return upload(rgb, self.device, timing=self) if self.device.type == "cuda" else rgb

    def detect_regions(self, rgb: np.ndarray, *, kind: str | None = None) -> list[Detection]:
        """Return padded xyxy boxes in source RGB coordinates without altering the pixels."""
        # Mobile-camera blur can extend beyond the predicted rectangle.
        return [Detection(detection.kind, (x1 - (x2 - x1) * self.box_padding, y1 - (y2 - y1) * self.box_padding,
                 x2 + (x2 - x1) * self.box_padding, y2 + (y2 - y1) * self.box_padding), detection.score)
                for detection in self.candidates(rgb, kind=kind) for x1, y1, x2, y2 in [detection.box]]

    def candidates(self, rgb: np.ndarray, *, kind: str | None = None) -> list[Detection]:
        """Return unpadded source boxes with type and score for cross-model confirmation."""
        if isinstance(rgb, torch.Tensor):
            rgb = self.prepare(rgb)
        with measure(self, "prepare-primary"):
            height, width = rgb.shape[:2]
            scale = min(1.0, self.max_side / max(height, width))
            detection_pixels = rgb
            if isinstance(rgb, torch.Tensor):
                from vision_shield.detection.cuda_pixels import resize

                with torch.cuda.stream(self.stream):
                    rgb.record_stream(self.stream)
                    if scale < 1:
                        detection_pixels = resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), "lanczos")
                    image = detection_pixels.flip(2).permute(2, 0, 1).contiguous()
            else:
                if scale < 1:
                    detection_pixels = np.asarray(Image.fromarray(rgb).resize(
                        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS))
                with measure(self, "input-copy"):
                    image = torch.from_numpy(detection_pixels[:, :, ::-1].transpose(2, 0, 1).copy()).to(self.device)
        # The released Gen1 model expects BGR uint8 CHW, without normalization.
        # 模型和输入必须在同一设备；后处理继续返回普通坐标。
        x_scale, y_scale = width / detection_pixels.shape[1], height / detection_pixels.shape[0]
        boxes = []
        # Gen1 JIT profiling adds seconds to cold frames without improving this CPU's steady rate.
        stream_context = torch.cuda.stream(self.stream) if isinstance(rgb, torch.Tensor) else nullcontext()
        with torch.inference_mode(), torch.jit.optimized_execution(False), stream_context:
            for model, model_kind, threshold in zip(self.models, ("face", "plate"), (self.face_threshold, 0.1)):
                # 局部复核只消费一种类别，跳过另一份独立模型而不改变目标模型输入。
                if model is None or kind is not None and kind != model_kind:
                    continue
                with measure(self, f"forward-primary-{model_kind}"):
                    sampler = getattr(self, "device_samples", None)
                    if sampler is not None and model_kind == "plate":
                        coordinates, _, scores, _ = sampler.forward(model, image, global_view=max(height, width) > self.max_side)
                    else:
                        coordinates, _, scores, _ = model(image)
                with measure(self, "post-primary"):
                    if (coordinates.ndim != 2 or coordinates.shape[1] != 4 or scores.shape != (len(coordinates),)
                            or not torch.isfinite(coordinates).all() or not torch.isfinite(scores).all()
                            or (scores < 0).any() or (scores > 1).any()
                            or (coordinates[:, 2:] <= coordinates[:, :2]).any()):
                        raise ValueError("Invalid detector output")
                    keep = torchvision.ops.nms(coordinates, scores, 0.3)
                    # Map each axis using the actual rounded input size, then mask full-resolution pixels.
                    selected = keep[scores[keep] > threshold]
                    with measure(self, "output-copy"):
                        values, confidences = coordinates[selected].cpu().tolist(), scores[selected].cpu().tolist()
                    boxes.extend(Detection(model_kind, (x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale), score)
                                 for (x1, y1, x2, y2), score in zip(values, confidences, strict=True))
        return boxes
