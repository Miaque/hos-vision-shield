"""Explicit P0 observers around real model/media calls; never enabled by default."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from functools import wraps
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from unittest.mock import patch
import sys


CONTEXT = ContextVar("video_profile_context", default={})
FORWARDS = tuple(f"forward-{model}-{kind}" for model in ("primary", "detail") for kind in ("face", "plate"))
MAX_MEASUREMENTS = 250_000


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


def runtime_metadata(detector, config):
    import cv2
    import torch
    import onnxruntime as ort

    if 'vision_shield.detection.yolo' in sys.modules:
        raise RuntimeError('B diagnostic must not load YOLO')
    device = detector.primary.device
    weights = {}
    for key in ('egoblur_face_model_path', 'egoblur_plate_model_path', 'face_model_path', 'plate_model_path'):
        with Path(config[key]).open('rb') as handle:
            weights[key] = hashlib.file_digest(handle, 'sha256').hexdigest()
    source = {}
    for name in ('detection.quality', 'detection.egoblur', 'detection.cuda_detail', 'detection.cpu',
                 'detection.host_profile', 'redaction', 'media.video'):
        with Path(sys.modules['vision_shield.' + name].__file__).open('rb') as handle:
            source[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(torch=torch.__version__, cuda=torch.version.cuda, ort=ort.__version__,
                gpu=torch.cuda.get_device_name(device),
                gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')),
                capability=list(torch.cuda.get_device_capability(device)),
                primary_parameter_dtypes=[sorted({str(p.dtype) for p in model.parameters()}) for model in detector.primary.models],
                primary_parameter_devices=[sorted({str(p.device) for p in model.parameters()}) for model in detector.primary.models],
                detail_providers=detector.detail.providers, torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                global_fp32_precision=torch.backends.fp32_precision,
                matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                cudnn_conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision,
                config={key: config[key] for key in ('detection_backend', 'inference_device', 'detection_max_side',
                                                     'detection_box_padding')} | {'redaction_mode': config.get('redaction_mode', 'solid')},
                weights_sha256=weights, source_sha256=source)


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
        self.measurements = 0

    @contextmanager
    def measure(self, name):
        with self.lock:
            if self.measurements >= MAX_MEASUREMENTS:
                raise RuntimeError("Host diagnostic measurement budget exceeded")
            self.measurements += 1
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
        self.device_samples.append(dict(stage=name, call=call, trace=target.name,
                                        context={key: value for key, value in CONTEXT.get().items() if key != "local_calls"},
                                        **summary))

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
