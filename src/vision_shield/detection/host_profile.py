"""Explicit P0 observers around real model/media calls; never enabled by default."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from unittest.mock import patch
import sys


CONTEXT = ContextVar("video_profile_context", default={})
FORWARDS = tuple(f"forward-{model}-{kind}" for model in ("primary", "detail") for kind in ("face", "plate"))
MAX_MEASUREMENTS = 250_000
LOCAL_FACE_SLOW_MS = 500
MAX_ORT_CALLS = 64


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
                    p95_ms=quantile(items, 0.95) if items else None,
                    p99_ms=quantile(items, 0.99) if items else None,
                    max_ms=max(items) if items else None,
                    over_500_ms=sum(value > 500 for value in items),
                    over_1000_ms=sum(value > 1000 for value in items))

    return dict(**summary(values), first_ms=values[0] if values else None,
                steady_after_10_calls=summary(values[10:]))


def attach_ort_nodes(samples, events):
    runs = sorted((event for event in events if event.get('name') == 'model_run'), key=lambda event: event['ts'])
    for sample in samples:
        run = runs[sample['call'] - 1]
        nodes = [event for event in events if event.get('cat') == 'Node'
                 and event['name'].endswith('_kernel_time')
                 and run['ts'] <= event['ts'] < run['ts'] + run['dur']]
        sample['ort_model_run_ms'] = run['dur'] / 1000
        sample['ort_top_nodes'] = [dict(name=event['name'][:256], host_ms=event['dur'] / 1000,
                                       op=event.get('args', {}).get('op_name'),
                                       provider=event.get('args', {}).get('provider'))
                                   for event in sorted(nodes, key=lambda event: event['dur'], reverse=True)[:8]]
    return [dict(call=index + 1, host_ms=event['dur'] / 1000) for index, event in enumerate(runs[:MAX_ORT_CALLS])]


def instrument_ort_nodes(model, collector):
    """A fresh, otherwise identical local session enables ORT's native node timer."""
    import onnxruntime as ort

    options = model.face.get_session_options()
    options.enable_profiling = True
    options.profile_file_prefix = str(collector.directory / 'ort-local-face')
    providers = [('CUDAExecutionProvider', model.face.get_provider_options()['CUDAExecutionProvider'])]
    model.face = ort.InferenceSession(model.face_model, sess_options=options, providers=providers)
    model.face.disable_fallback()
    collector.ort_session = model.face


def runtime_metadata(detector, config):
    import cv2
    import torch
    import onnxruntime as ort
    from vision_shield.detection import cuda_pixels  # Resolve the actual helper before its first forward.

    if config['detection_backend'] != 'quality_yolo':
        raise RuntimeError('A diagnostic requires quality_yolo')
    device = detector.primary.device
    weights = {}
    for key in ('egoblur_plate_model_path', 'face_model_path', 'plate_model_path'):
        with Path(config[key]).open('rb') as handle:
            weights[key] = hashlib.file_digest(handle, 'sha256').hexdigest()
    with detector.primary.face_path.open('rb') as handle:
        weights['primary_face'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    source = {}
    for name in ('detection.quality', 'detection.plate_readability', 'detection.egoblur', 'detection.cuda_detail', 'detection.cpu',
                 'detection.host_profile', 'detection.yolo', 'detection.cuda_pixels', 'redaction', 'media.video',
                 'media.plate_lookahead', 'media.streams'):
        with Path(sys.modules['vision_shield.' + name].__file__).open('rb') as handle:
            source[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(torch=torch.__version__, cuda=torch.version.cuda, ort=ort.__version__,
                gpu=torch.cuda.get_device_name(device),
                gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')),
                capability=list(torch.cuda.get_device_capability(device)),
                primary_parameter_dtypes=[sorted({str(p.dtype) for p in model.parameters()}) if model is not None else None for model in detector.primary.models],
                primary_parameter_devices=[sorted({str(p.device) for p in model.parameters()}) if model is not None else None for model in detector.primary.models],
                primary_face_providers=detector.primary.providers,
                primary_face_input_types=sorted({session.get_inputs()[0].type for session in detector.primary.sessions.values()}),
                detail_providers=detector.detail.providers,
                detail_face_provider_options=detector.detail.face.get_provider_options(),
                torch_cudnn_version=torch.backends.cudnn.version(),
                torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                global_fp32_precision=torch.backends.fp32_precision,
                matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                cudnn_conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision,
                config={key: config[key] for key in ('detection_backend', 'inference_device', 'detection_max_side',
                                                     'detection_box_padding')} | {'redaction_mode': config.get('redaction_mode', 'solid'),
                                                    'detection_interval': config.get('detection_interval', 1),
                                                    'tracking': config.get('tracking', 'template')},
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
        self.candidates, self.candidate_bytes, self.candidates_truncated = [], 0, False
        self.masks, self.mask_bytes, self.masks_truncated, self.mask_shape = [], 0, False, None
        self.local_requests, self.local_sessions, self.local_shapes, self.local_longest = 0, {}, {}, []
        self.local_last_shape = {}
        self.ort_session = None
        self.ort_runs = []

    def record_masks(self, shape, boxes):
        height, width = shape[:2]
        rectangles = [[max(0, math.floor(x1)), max(0, math.floor(y1)),
                       min(width, math.ceil(x2)), min(height, math.ceil(y2))] for x1, y1, x2, y2 in boxes]
        row = [CONTEXT.get()["frame"], [box for box in rectangles if box[2] > box[0] and box[3] > box[1]]]
        size = len(json.dumps(row).encode("utf-8"))
        with self.lock:
            self.mask_shape = [height, width]
            if self.mask_bytes + size > 64 * 1024:
                self.masks_truncated = True
                return
            self.masks.append(row)
            self.mask_bytes += size

    def record_candidates(self, stage, detections):
        context = CONTEXT.get()
        if context.get("purpose") != "discovery" or not 1 <= context.get("frame", 0) <= 32:
            return
        row = dict(stage=stage, frame=context["frame"], source_seconds=context.get("source_seconds"),
                   detections=[dict(kind=d.kind, box=[float(v) for v in d.box], score=float(d.score)) for d in detections])
        size = len(json.dumps(row).encode("utf-8"))
        with self.lock:
            # Keep the diagnostic event within the child pipe limit; short clips target later source frames.
            if self.candidate_bytes + size > 64 * 1024:
                self.candidates_truncated = True
                return
            self.candidates.append(row)
            self.candidate_bytes += size

    @contextmanager
    def measure(self, name):
        with self.lock:
            if self.measurements >= MAX_MEASUREMENTS:
                raise RuntimeError("Host diagnostic measurement budget exceeded")
            self.measurements += 1
        metadata = {key: value for key, value in CONTEXT.get().items() if key not in ("local_calls", "local_details")}
        selected, call = False, None
        if name in (*FORWARDS, 'ort-local-face'):
            with self.lock:
                self.forward_counts[name] += 1
                call = self.forward_counts[name]
            selected = (name == self.device_stage and call in self.checkpoints
                        or name == 'ort-local-face' and self.device_stage == 'local-face'
                        and len(self.device_samples) < len(self.checkpoints))
        started, failed = self.clock(), False
        try:
            with self.trace(name, call, slow_only=name == 'ort-local-face') if selected else nullcontext():
                yield
        except BaseException:
            failed = True
            raise
        finally:
            milliseconds = (self.clock() - started) * 1000
            metadata.update(CONTEXT.get().get('local_details', {}))
            with self.lock:
                self.samples[name].append(milliseconds)
                self.seconds[name] += milliseconds / 1000
                self.groups[(name, metadata.get("origin", "pipeline"), metadata.get("purpose", "pipeline"),
                             metadata.get("view", "none"))].append(milliseconds)
                self.failures[name] += failed
                if name in FORWARDS or metadata.get('local_id'):
                    self.file.write(json.dumps(dict(stage=name, call=call, host_ms=milliseconds,
                                                    failed=failed, device_sampled=selected, **metadata)) + "\n")
                if name == 'ort-local-face':
                    self.local_longest.append(dict(host_ms=milliseconds, call=call, **metadata))
                    self.local_longest.sort(key=lambda row: row['host_ms'], reverse=True)
                    del self.local_longest[8:]

    @contextmanager
    def trace(self, name, call, *, slow_only=False):
        from torch.profiler import ProfilerActivity, profile, record_function
        from vision_shield.detection.device_profile import MAX_TRACE_BYTES, summarize

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with record_function(name):
                yield
        if slow_only:
            elapsed = next(event.cpu_time_total for event in profiler.events() if event.name == name) / 1000
            if elapsed <= LOCAL_FACE_SLOW_MS:
                return
        slot = self.checkpoints[len(self.device_samples)] if slow_only else call
        target = self.directory / f"{'local-face' if slow_only else name}-{slot}.json"
        profiler.export_chrome_trace(str(target))
        if target.stat().st_size > MAX_TRACE_BYTES:
            target.unlink()
            raise RuntimeError("Device trace exceeded 32 MiB budget")
        with patch("vision_shield.detection.device_profile.SCOPE", name):
            summary = summarize(json.loads(target.read_text("utf-8")))
        self.device_samples.append(dict(stage=name, call=call, sample=slot, trace=target.name,
                                        context={key: value for key, value in CONTEXT.get().items()
                                                 if key not in ('local_calls', 'local_details')}
                                                | CONTEXT.get().get('local_details', {}),
                                        **summary))

    def snapshot(self):
        from vision_shield.detection.timing import STAGES

        with self.lock:
            return {name: seconds for name, seconds in self.seconds.items() if name in STAGES}

    def report(self, frames):
        self.finish_ort()
        models = {name: dict(**statistics(self.samples[name]), failures=self.failures[name],
                             calls_per_frame=len(self.samples[name]) / frames if frames else None) for name in FORWARDS}
        return dict(models=models, host_stages={name: statistics(values) for name, values in self.samples.items()},
                    host_groups=[dict(stage=name, origin=origin, purpose=purpose, view=view, **statistics(values))
                                 for (name, origin, purpose, view), values in sorted(self.groups.items())],
                    views={name: dict(counts) for name, counts in self.views.items()}, device_samples=self.device_samples,
                    local_faces=dict(timing=statistics(self.samples['ort-local-face']), longest=self.local_longest),
                    ort_local_runs=self.ort_runs,
                    candidate_evidence=dict(first_frames=32, byte_limit=64 * 1024,
                                            truncated=self.candidates_truncated, rows=self.candidates),
                    mask_evidence=dict(shape=self.mask_shape, byte_limit=64 * 1024,
                                       truncated=self.masks_truncated, rows=self.masks),
                    notes=["Host-wall scopes include submission/waits; they are not GPU kernel time.",
                           "Nested/concurrent stages must not be summed into total time.",
                           "Steady means calls after the first 10 per group; new input shapes may still initialize.",
                           "ORT node timing covers only the first 64 local calls; later CUDA samples may have no node detail.",
                           "Device traces are a separate instrumented run; its FPS is not acceptance throughput."])

    def close(self):
        try:
            self.finish_ort()
        finally:
            self.file.close()

    def finish_ort(self):
        if self.ort_session is None:
            return
        session, self.ort_session = self.ort_session, None
        target = Path(session.end_profiling())
        if target.stat().st_size > 32 * 1024**2:
            target.unlink()
            raise RuntimeError('ORT trace exceeded 32 MiB budget')
        self.ort_runs = attach_ort_nodes(self.device_samples, json.loads(target.read_text('utf-8')))


def instrument_local_faces(stack, collector):
    """Observe existing local work only inside the opt-in, isolated diagnostic process."""
    import cv2
    import onnxruntime as ort
    import torch

    def tensor_method(name, stage):
        original = getattr(torch.Tensor, name)

        @wraps(original)
        def observed(value, *args, **kwargs):
            if not CONTEXT.get().get('local_id'):
                return original(value, *args, **kwargs)
            if name == 'cpu':
                CONTEXT.get()['local_details']['readback_source_device'] = str(value.device)
            with collector.measure(stage):
                return original(value, *args, **kwargs)

        stack.enter_context(patch.object(torch.Tensor, name, observed))

    tensor_method('cpu', 'local-readback')
    tensor_method('numpy', 'local-numpy')
    original_resize = cv2.resize

    def resize(pixels, *args, **kwargs):
        details = CONTEXT.get().get('local_details')
        if details is None:
            return original_resize(pixels, *args, **kwargs)
        first_resize = 'roi_shape' not in details
        if first_resize:
            details['roi_shape'] = list(pixels.shape)
        with collector.measure('local-resize'):
            output = original_resize(pixels, *args, **kwargs)
        if first_resize:
            details['resized_shape'] = list(output.shape)
        return output

    stack.enter_context(patch.object(cv2, 'resize', resize))
    original_run = ort.InferenceSession.run

    def run(session, outputs, feed, *args, **kwargs):
        context = CONTEXT.get()
        if not context.get('local_id') or not context.get('local_face'):
            return original_run(session, outputs, feed, *args, **kwargs)
        shape = tuple(next(iter(feed.values())).shape)
        identity = id(session)
        instance = collector.local_sessions.setdefault(identity, len(collector.local_sessions) + 1)
        seen = collector.local_shapes.setdefault(identity, set())
        first = shape not in seen
        seen.add(shape)
        previous = collector.local_last_shape.get(identity)
        collector.local_last_shape[identity] = shape
        with scope(input_shape=list(shape), session_instance=instance,
                   provider=session.get_providers()[0], first_local_shape=first,
                   shape_changed=previous is not None and previous != shape,
                   input_dtype=str(next(iter(feed.values())).dtype)), collector.measure('ort-local-face'):
            result = original_run(session, outputs, feed, *args, **kwargs)
        if collector.ort_session is session and collector.forward_counts['ort-local-face'] >= MAX_ORT_CALLS:
            collector.finish_ort()
        return result

    stack.enter_context(patch.object(ort.InferenceSession, 'run', run))


def instrument_candidates(model, name, collector):
    original = model.candidates

    @wraps(original)
    def candidates(pixels, *args, **kwargs):
        calls = CONTEXT.get().get("local_calls")
        mirror = False
        if calls is not None:
            calls[name] = calls.get(name, 0) + 1
            mirror = calls[name] > 1
        batched_views = (hasattr(model, "_plate_session") and getattr(pixels, "is_cuda", False)
                         and model.parallel and kwargs.get("kind") != "face"
                         and max(pixels.shape[:2]) > model.tile_side)
        regions = list(original_views(*pixels.shape[:2][::-1], model.tile_side)) if batched_views else None
        mixed = {"source_region": None, "tile_index": None} if batched_views else {}
        # Collection leaves the final tile scope active; a batch cannot inherit that tile's identity.
        with scope(model_group=name, source_shape=list(pixels.shape), mirror=mirror,
                   view="mixed" if batched_views else "global", batched_candidate_regions=regions, **mixed):
            with collector.measure(f"candidate-{name}"):
                result = original(pixels, *args, **kwargs)
            collector.record_candidates(name, result)
            return result

    model.candidates = candidates
    if hasattr(model, "_views"):
        original_views = model._views

        def views(*args, **kwargs):
            for index, box in enumerate(original_views(*args, **kwargs)):
                view = "global" if index == 0 else "tile"
                with collector.lock:
                    collector.views[CONTEXT.get().get("purpose", "discovery")][view] += 1
                mixed = bool(CONTEXT.get().get("batched_candidate_regions"))
                with scope(view="mixed" if mixed else view, tile_index=None if mixed else index,
                           source_region=None if mixed else list(box)):
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
    from vision_shield.detection import cpu, cuda_detail, quality, yolo
    from vision_shield.redaction import FrameRedactor

    instrument_local_faces(stack, collector)

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
                with collector.lock:
                    collector.local_requests += 1
                    request = collector.local_requests
                target = getattr(args[1], 'box', args[1])
                metadata.update(local_id=f"{CONTEXT.get().get('frame', 0)}:{request}",
                                parent_local_id=CONTEXT.get().get('local_id'),
                                task_id=collector.directory.parent.name,
                                target_box=list(target), local_details={})
                if purpose == "track-confirm":
                    metadata["origin"] = purpose
            with scope(**metadata), collector.measure(stage):
                result = original(instance, *args, **kwargs)
                if stage == "quality-detection":
                    collector.record_candidates("accepted", result)
                if frame and not getattr(instance, '_plate_template_fallback', False):
                    collector.record_masks(args[0].shape, result)
                return result

        stack.enter_context(patch.object(owner, name, wrapped))

    method(yolo.Detector, "__init__", "initialize-primary")
    method(cuda_detail.Detector, "__init__", "initialize-detail")
    original_faces = cuda_detail.Detector._faces

    def faces(instance, pixels):
        with scope(local_face=True):
            return original_faces(instance, pixels)

    stack.enter_context(patch.object(cuda_detail.Detector, '_faces', faces))
    method(quality.Detector, "detect_regions", "quality-detection")
    method(quality.Detector, "_confirmed", "confirm-request", purpose="local-confirm")
    method(quality.Detector, "_recover_face", "recover-request", purpose="local-enlarged")
    method(FrameRedactor, "regions", "frame-regions", frame=True)
    method(FrameRedactor, "_detections", "detect-pass")
    method(FrameRedactor, "_recheck", "track-recheck-request", purpose="track-confirm")
    method(FrameRedactor, "_follow", "template-follow")
