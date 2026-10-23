"""Opt-in P0 diagnostics around the unchanged production video/detector call chain.

Use PYTHONPATH to select the fixed route's src directory. Host/device diagnostic
FPS is not acceptance throughput. This script never connects to a server.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from functools import wraps
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
from time import perf_counter
from unittest.mock import patch


CONTEXT = ContextVar("video_profile_context", default={})
FORWARDS = tuple(f"forward-{model}-{kind}" for model in ("primary", "detail") for kind in ("face", "plate"))


@contextmanager
def scope(**values):
    token = CONTEXT.set({**CONTEXT.get(), **values})
    try:
        yield
    finally:
        CONTEXT.reset(token)


class ContextPool(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)


def statistics(values):
    def quantile(items, fraction):
        ordered = sorted(items)
        index = (len(ordered) - 1) * fraction
        left = int(index)
        return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (index - left)

    def summary(items):
        return dict(calls=len(items), cumulative_ms=sum(items),
                    p50_ms=quantile(items, 0.5) if items else None,
                    p95_ms=quantile(items, 0.95) if items else None)

    return dict(**summary(values), first_ms=values[0] if values else None,
                steady_after_10_calls=summary(values[10:]))


class ProfileTimes:
    def __init__(self, directory, *, clock=perf_counter, device_stage=None, checkpoints=(64, 424, 768)):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = (self.directory / "calls.jsonl").open("x", encoding="utf-8")
        self.clock, self.device_stage, self.checkpoints = clock, device_stage, checkpoints
        self.lock = Lock()
        self.samples, self.groups = defaultdict(list), defaultdict(list)
        self.seconds = defaultdict(float)
        self.failures, self.forward_counts = Counter(), Counter()
        self.views, self.device_samples = defaultdict(Counter), []
        self.frames = 0

    @contextmanager
    def measure(self, name):
        metadata = {key: value for key, value in CONTEXT.get().items() if key != "local_calls"}
        selected = False
        if name in FORWARDS:
            with self.lock:
                self.forward_counts[name] += 1
                call = self.forward_counts[name]
            selected = name == self.device_stage and call in self.checkpoints
        started, failed = self.clock(), False
        try:
            with self.trace(name, call) if selected else nullcontext():
                yield
        except BaseException:
            failed = True
            raise
        finally:
            milliseconds = (self.clock() - started) * 1000
            with self.lock:
                self.samples[name].append(milliseconds)
                self.seconds[name] += milliseconds / 1000
                self.groups[(name, metadata.get("origin", "pipeline"), metadata.get("purpose", "pipeline"),
                             metadata.get("view", "none"))].append(milliseconds)
                self.failures[name] += failed
                if name in FORWARDS:
                    self.file.write(json.dumps(dict(stage=name, call=call, host_ms=milliseconds,
                                                    failed=failed, device_sampled=selected, **metadata)) + "\n")

    @contextmanager
    def trace(self, name, call):
        from torch.profiler import ProfilerActivity, profile, record_function
        from vision_shield.detection.device_profile import MAX_TRACE_BYTES, summarize

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with record_function(name):
                yield
        target = self.directory / f"{name}-{call}.json"
        profiler.export_chrome_trace(str(target))
        if target.stat().st_size > MAX_TRACE_BYTES:
            target.unlink()
            raise RuntimeError("Device trace exceeded 32 MiB budget")
        with patch("vision_shield.detection.device_profile.SCOPE", name):
            summary = summarize(json.loads(target.read_text("utf-8")))
        self.device_samples.append(dict(stage=name, call=call, trace=target.name, **summary))

    def snapshot(self):
        from vision_shield.detection.timing import STAGES

        with self.lock:
            return {name: seconds for name, seconds in self.seconds.items() if name in STAGES}

    def report(self, frames):
        models = {name: dict(**statistics(self.samples[name]), failures=self.failures[name],
                             calls_per_frame=len(self.samples[name]) / frames if frames else None) for name in FORWARDS}
        return dict(models=models, host_stages={name: statistics(values) for name, values in self.samples.items()},
                    host_groups=[dict(stage=name, origin=origin, purpose=purpose, view=view, **statistics(values))
                                 for (name, origin, purpose, view), values in sorted(self.groups.items())],
                    views={name: dict(counts) for name, counts in self.views.items()}, device_samples=self.device_samples,
                    notes=["Host-wall scopes include submission/waits; they are not GPU kernel time.",
                           "Nested/concurrent stages must not be summed into total time.",
                           "Steady means calls after the first 10 per group; new input shapes may still initialize.",
                           "Device traces are a separate instrumented run; its FPS is not acceptance throughput."])

    def close(self):
        self.file.close()


def instrument_candidates(model, name, collector):
    original = model.candidates

    @wraps(original)
    def candidates(pixels, *args, **kwargs):
        calls = CONTEXT.get().get("local_calls")
        mirror = False
        if calls is not None:
            calls[name] = calls.get(name, 0) + 1
            mirror = calls[name] > 1
        with scope(model_group=name, source_shape=list(pixels.shape), mirror=mirror, view="global"):
            with collector.measure(f"candidate-{name}"):
                return original(pixels, *args, **kwargs)

    model.candidates = candidates
    if hasattr(model, "_views"):
        original_views = model._views

        def views(*args, **kwargs):
            for index, box in enumerate(original_views(*args, **kwargs)):
                view = "global" if index == 0 else "tile"
                with collector.lock:
                    collector.views[CONTEXT.get().get("purpose", "discovery")][view] += 1
                with scope(view=view, tile_index=index, source_region=list(box)):
                    yield box

        model._views = views


class TimedPacket:
    def __init__(self, packet, collector):
        self.packet, self.collector = packet, collector

    def __getattr__(self, name):
        return getattr(self.packet, name)

    def decode(self):
        with self.collector.measure("decode-host"):
            return self.packet.decode()


class TimedEncoder:
    def __init__(self, stream, collector):
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "collector", collector)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __setattr__(self, name, value):
        setattr(self.stream, name, value)

    def encode(self, *args):
        with self.collector.measure("encode-host"):
            return self.stream.encode(*args)


class TimedContainer:
    def __init__(self, container, collector):
        self.container, self.collector = container, collector

    def __getattr__(self, name):
        return getattr(self.container, name)

    def __enter__(self):
        self.container.__enter__()
        return self

    def __exit__(self, *args):
        with self.collector.measure("container-close"):
            return self.container.__exit__(*args)

    def demux(self):
        packets = iter(self.container.demux())
        while True:
            with self.collector.measure("demux-read"):
                packet = next(packets, None)
            if packet is None:
                return
            yield TimedPacket(packet, self.collector) if packet.stream.type == "video" else packet

    def add_stream(self, *args, **kwargs):
        with self.collector.measure("encoder-initialize"):
            return TimedEncoder(self.container.add_stream(*args, **kwargs), self.collector)

    def mux(self, packet):
        with self.collector.measure("mux-host"):
            return self.container.mux(packet)


def instrument(stack, collector):
    import av
    from vision_shield.detection import cpu, cuda_detail, egoblur, quality
    from vision_shield.redaction import FrameRedactor

    original_open = av.open
    stack.enter_context(patch.object(av, "open", lambda *args, **kwargs: TimedContainer(original_open(*args, **kwargs), collector)))
    for module in (quality, cpu):
        stack.enter_context(patch.object(module, "ThreadPoolExecutor", ContextPool))
    stack.enter_context(patch.object(quality, "StageTimes", lambda: collector))

    def method(owner, name, stage, *, purpose=None, frame=False):
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(instance, *args, **kwargs):
            metadata = {}
            if frame:
                collector.frames += 1
                metadata = dict(frame=collector.frames, source_seconds=args[1], origin="discovery", purpose="discovery")
            elif purpose:
                metadata = dict(purpose=purpose, local_calls={})
                if purpose == "track-confirm":
                    metadata["origin"] = purpose
            with scope(**metadata), collector.measure(stage):
                return original(instance, *args, **kwargs)

        stack.enter_context(patch.object(owner, name, wrapped))

    method(egoblur.Detector, "__init__", "initialize-primary")
    method(cuda_detail.Detector, "__init__", "initialize-detail")
    method(quality.Detector, "detect_regions", "quality-detection")
    method(quality.Detector, "_confirmed", "confirm-request", purpose="local-confirm")
    method(quality.Detector, "_recover_face", "recover-request", purpose="local-enlarged")
    method(FrameRedactor, "regions", "frame-regions", frame=True)
    method(FrameRedactor, "_detections", "detect-pass")
    method(FrameRedactor, "_recheck", "track-recheck-request", purpose="track-confirm")
    method(FrameRedactor, "_follow", "template-follow")


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
                  harness_sha256=sha256(__file__),
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
