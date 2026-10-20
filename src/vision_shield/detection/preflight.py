"""离线核验模型身份并实际推理；不连接业务服务或下载权重。"""

import hashlib
import json
import os
from pathlib import Path
import re


class PreflightError(RuntimeError):
    """仅携带文件、角色和阶段的有限启动诊断。"""


def verify_models(config: dict) -> list[dict]:
    """按后端选择既有清单，逐个流式校验权重哈希和可读权限。"""
    backend = config["detection_backend"]
    selected = []
    if backend in {"quality", "cpu"}:
        selected += [("YuNet人脸", "face_model_path", "cpu-manifest.json", "yunet_face.onnx"),
                     ("HyperLPR车牌", "plate_model_path", "cpu-manifest.json", "hyperlpr_plate.onnx")]
    if backend == "yolo":
        selected += [("YOLO人脸", "face_model_path", "yolo-manifest.json", "yolov8n-face.onnx"),
                     ("YOLO车牌", "plate_model_path", "yolo-manifest.json", "yolov11s-plate.onnx")]
    if backend in {"quality", "egoblur"}:
        face_key, plate_key = ("egoblur_face_model_path", "egoblur_plate_model_path") if backend == "quality" else ("face_model_path", "plate_model_path")
        selected += [("EgoBlur人脸", face_key, "manifest.json", "face.jit"),
                     ("EgoBlur车牌", plate_key, "manifest.json", "license_plate.jit")]
    if not selected:
        raise PreflightError("模型预检失败：不支持的检测后端")
    verified = []
    for role, key, manifest_name, entry_name in selected:
        path = Path(config[key])
        stage = "清单"
        try:
            manifest = json.loads((path.parent / manifest_name).read_text("utf-8"))
            expected = next(model["sha256"] for model in manifest["models"] if model["path"] == entry_name)
            stage = "读取与哈希"
            if not path.is_file() or not path.stat().st_mode & 0o444:
                raise PermissionError
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            raise PreflightError(f"模型预检失败：角色={role} 文件={path.name} 阶段={stage}") from None
        verified.append(dict(role=role, file=path.name, sha256=actual))
    return verified


def validate_device(device: str) -> None:
    """部署标识约束可见设备；原生运行不新增单卡限制。"""
    if device not in {"cpu", "cuda"}:
        raise PreflightError("模型预检失败：设备仅支持 cpu/cuda")
    if device == "cpu":
        return
    identifier = os.environ.get("GPU_DEVICE_ID")
    if identifier is not None and not re.fullmatch(
        r"(?:0|[1-9][0-9]*|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", identifier
    ):
        raise PreflightError("模型预检失败：GPU_DEVICE_ID 必须是单个卡号或完整 GPU UUID")
    import torch
    if not torch.cuda.is_available():
        raise PreflightError("模型预检失败：CUDA 不可用")
    if identifier is not None and torch.cuda.device_count() != 1:
        raise PreflightError("模型预检失败：部署必须仅有唯一 CUDA 可见设备")


def run_inference(config: dict) -> dict:
    """复用工厂实际推理合成帧，并执行非空 NMS 验证目标设备算子。"""
    import numpy as np
    import torch
    import torchvision
    from vision_shield.detection import load_detector

    device = config.get("inference_device", "cpu")
    validate_device(device)
    detector = load_detector(config)
    if config["detection_backend"] == "yolo":
        # YOLO 持有 ORT 会话，不能按 PyTorch 参数读取设备。
        if device != "cpu" or len(detector.models) != 2 or any(
            model.get_providers() != ["CPUExecutionProvider"] for model in detector.models
        ):
            raise PreflightError("模型预检失败：YOLO 必须仅使用 CPUExecutionProvider")
        model_devices = ["cpu"]
    else:
        primary = getattr(detector, "primary", detector)
        model_devices = sorted({tensor.device.type for model in getattr(primary, "models", [])
                                for tensor in (*model.parameters(), *model.buffers())})
    if config["detection_backend"] != "cpu" and model_devices != [device]:
        raise PreflightError("模型预检失败：模型与请求设备不一致")
    detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    kept = torchvision.ops.nms(boxes, scores, 0.3)
    if kept.numel() == 0 or kept.device.type != device:
        raise PreflightError("模型预检失败：NMS 未在请求设备执行")
    report = dict(backend=config["detection_backend"], device=device, model_devices=model_devices,
                  nms_device=kept.device.type, torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda_build=torch.version.cuda)
    if device == "cuda":
        report.update(gpu_name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      architectures=torch.cuda.get_arch_list(), visible_devices=torch.cuda.device_count(),
                      selected_gpu=os.environ.get("GPU_DEVICE_ID"))
    return report


def run_preflight(config: dict) -> dict:
    """身份核验成功后才允许加载；不传播底层异常中的地址或凭据。"""
    models = verify_models(config)
    try:
        return {**run_inference(config), "models": models}
    except PreflightError:
        raise
    except Exception:
        raise PreflightError("模型预检失败：阶段=加载与推理") from None


def main() -> None:
    """独立容器运维入口，失败返回非零退出码。"""
    from vision_shield.config import Settings
    try:
        report = run_preflight(Settings().model_dump(mode="json"))
    except (PreflightError, ValueError) as error:
        # 配置校验也不输出原始环境变量，避免敏感地址进入终端记录。
        message = str(error) if isinstance(error, PreflightError) else "模型预检失败：配置无效"
        print(json.dumps({"error": message}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
