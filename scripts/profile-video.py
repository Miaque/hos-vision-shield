"""Opt-in P0 diagnostics around the unchanged production video/detector call chain.

Use PYTHONPATH to select the fixed route's src directory. Host/device diagnostic
FPS is not acceptance throughput. This script never connects to a server.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import perf_counter


# Load this worktree's observer while PYTHONPATH may point at the fixed A source.
observer_path = Path(__file__).resolve().parents[1] / "src/vision_shield/detection/host_profile.py"
observer_spec = importlib.util.spec_from_file_location("video_profile_observer", observer_path)
observer = importlib.util.module_from_spec(observer_spec)
observer_spec.loader.exec_module(observer)
FORWARDS = observer.FORWARDS
ProfileTimes = observer.ProfileTimes
ContextPool = observer.ContextPool
scope = observer.scope
statistics = observer.statistics
instrument = observer.instrument
instrument_candidates = observer.instrument_candidates
TimedContainer = observer.TimedContainer


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("A", "B"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("host", "device", "throughput"), default="host")
    parser.add_argument("--device-stage", choices=FORWARDS, default="forward-primary-plate")
    parser.add_argument("--checkpoints", default="64,424,768")
    args = parser.parse_args()
    if args.target.exists() or args.report_dir.exists():
        parser.error("Use new target/report paths; previous evidence is never overwritten")
    if args.source.resolve() == args.target.resolve():
        parser.error("Source and target must differ")
    checkpoints = tuple(int(value) for value in args.checkpoints.split(","))
    if not checkpoints or min(checkpoints) < 1 or len(set(checkpoints)) != len(checkpoints):
        parser.error("Checkpoints must be unique positive forward call indices")
    args.report_dir.mkdir(parents=True)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    import torch
    import onnxruntime as ort
    import vision_shield
    from vision_shield.config import Settings
    from vision_shield.detection import load_detector
    from vision_shield.media.video import process_video

    root = Path(vision_shield.__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.route == "A":
        fixed = "581c175a0fad21efe10a0caf08a01e37b25fd3ad"
        subprocess.run(["git", "diff", "--exit-code", "--quiet", fixed, "--", "src/vision_shield"], cwd=root, check=True)
    config = Settings(_env_file=None, detection_backend="quality" if args.route == "B" else "quality_yolo",
                      inference_device="cuda", detection_max_side=1280, detection_box_padding=0.15,
                      face_model_path=args.model_dir / "yunet_face.onnx", plate_model_path=args.model_dir / "hyperlpr_plate.onnx",
                      egoblur_face_model_path=args.model_dir / "face.jit",
                      egoblur_plate_model_path=args.model_dir / "license_plate.jit").model_dump()
    config.update(redaction_mode="solid", detection_interval=3 if args.route == "A" else 1,
                  tracking="flow" if args.route == "A" else "template")
    collector = None if args.mode == "throughput" else ProfileTimes(
        args.report_dir, device_stage=args.device_stage if args.mode == "device" else None, checkpoints=checkpoints)
    result = dict(state="running", route=args.route, mode=args.mode, git_commit=head, source_root=str(root),
                  harness_sha256=sha256(__file__), observer_sha256=sha256(observer_path),
                  config={key: str(config[key]) for key in ("detection_backend", "inference_device", "detection_max_side",
                          "detection_box_padding", "redaction_mode", "detection_interval", "tracking")},
                  source_sha256=sha256(args.source), weights={}, instrumented=args.mode != "throughput",
                  source_files={str(path.relative_to(root)): sha256(path) for path in sorted((root / "src/vision_shield").rglob("*.py"))},
                  dirty_paths=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines())
    for key in ("face_model_path", "plate_model_path", "egoblur_plate_model_path"):
        result["weights"][key] = dict(path=str(config[key]), sha256=sha256(config[key]))
    face = config["egoblur_face_model_path"] if args.route == "B" else root / "src/vision_shield/detection/assets/yolov8n_face.onnx"
    result["weights"]["primary_face"] = dict(path=str(face), sha256=sha256(face))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        result["runtime"] = dict(torch=torch.__version__, cuda=torch.version.cuda, ort=ort.__version__,
                                 gpu=torch.cuda.get_device_name(), gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                                 capability=list(torch.cuda.get_device_capability()),
                                 global_fp32_precision=torch.backends.fp32_precision,
                                 matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                                 cudnn_conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision)
        with ExitStack() as stack:
            if collector:
                instrument(stack, collector)
            initialize = perf_counter()
            detector = load_detector(config)
            result["initialize_seconds"] = perf_counter() - initialize
            import cv2
            result["runtime"].update(torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads())
            if args.route == "B" and "vision_shield.detection.yolo" in sys.modules:
                raise RuntimeError("Route B must not import YOLO")
            result["runtime"]["primary_parameter_dtypes"] = [sorted({str(p.dtype) for p in model.parameters()})
                                                             if model is not None else None for model in detector.primary.models]
            result["runtime"]["primary_parameter_devices"] = [sorted({str(p.device) for p in model.parameters()})
                                                              if model is not None else None for model in detector.primary.models]
            result["runtime"]["detail_providers"] = detector.detail.providers
            if collector:
                instrument_candidates(detector.primary, "primary", collector)
                instrument_candidates(detector.detail, "detail", collector)
            latest = {}

            def emit(event):
                latest.update(event)
                if event.get("stats", {}).get("processed_frames", 1) % 100 == 0:
                    print(json.dumps(event["stats"]), flush=True)

            processing = perf_counter()
            frames = process_video(args.source, args.target, detector, config, Event(), emit)
            result.update(state="complete", frames=frames, processing_seconds=perf_counter() - processing, event=latest)
            result["media_fps"] = frames / latest["stats"]["elapsed_seconds"]
            result["measurement_start_to_media_finish_fps"] = frames / (perf_counter() - started)
            result["measurement_boundary"] = "Before framework imports to process_video return; excludes interpreter startup, output verification and report writing."
            result["task_creation_to_finish_fps"] = None  # Direct CLI has no API/queue boundary.
            result["peak_torch_memory_bytes"] = torch.cuda.max_memory_allocated()
            if args.route == "B" and "vision_shield.detection.yolo" in sys.modules:
                raise RuntimeError("Route B imported YOLO during processing")
            result["output_sha256"] = sha256(args.target)
    except BaseException as error:
        result.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        if collector:
            collector.close()
            result["profile"] = collector.report(collector.frames)
        (args.report_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: result[key] for key in ("state", "route", "mode", "frames", "media_fps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
