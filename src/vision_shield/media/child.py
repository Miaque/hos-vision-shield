"""Isolated or resident media execution; parent-pipe EOF revokes either mode."""

import json
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import time
import traceback
from contextlib import ExitStack

from av.error import FFmpegError
from vision_shield.media.attempts import failure_details, model_runtime_failure

# PyAV ArgumentError is a ValueError; a disrupted mux still needs a fresh session.
RETRYABLE_STREAM_ERRORS = (OSError, BufferError, FFmpegError)


def emit(event):
    print(json.dumps(event), flush=True)


def process(payload, stop, detector=None, *, on_initialized=None):
    observers = ExitStack()
    collector = None

    try:
        cfg = payload["config"]
        if payload["kind"] == "probe":
            from vision_shield.detection.preflight import run_preflight
            report = run_preflight(cfg)
            if on_initialized:
                on_initialized()
            if not stop.is_set():
                emit({"probe": report})
            return 0, True
        from vision_shield.detection import load_detector
        from vision_shield.media.images import process_image
        from vision_shield.media.video import process_video
        from vision_shield.media.streams import process_stream
        from PIL import Image

        directory = Path(payload["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        profile_mode = payload.get('profile_detection')
        if isinstance(profile_mode, str):
            from vision_shield.detection.host_profile import FORWARDS, ProfileTimes, instrument

            if (profile_mode not in ('host', *FORWARDS) or cfg['inference_device'] != 'cuda'
                    or cfg['detection_backend'] != 'quality_yolo'):
                raise ValueError('A model profiling requires quality_yolo/CUDA')
            collector = ProfileTimes(directory / 'host-profile',
                                     device_stage=None if profile_mode == 'host' else profile_mode)
            observers.callback(collector.close)
            instrument(observers, collector)
        loading = time.monotonic()
        reused = detector is not None
        if detector is None:
            detector = load_detector(cfg)
        else:
            for owner in (detector, getattr(detector, 'primary', None), getattr(detector, 'detail', None)):
                timing = getattr(owner, 'stage_times', None)
                if timing is not None:
                    timing.reset()
                for name, value in (('candidate_timings', {}), ('_proposal_candidates', None)):
                    if hasattr(owner, name):
                        setattr(owner, name, value)
        Image.init()
        if on_initialized:
            on_initialized()
        load_seconds = 0.0 if reused else time.monotonic() - loading
        emit({"model_load_seconds": load_seconds, "model_reused": reused})
        if collector:
            from vision_shield.detection.host_profile import instrument_candidates, runtime_metadata

            instrument_candidates(detector.primary, 'primary', collector)
            instrument_candidates(detector.detail, 'detail', collector)
            metadata = runtime_metadata(detector, cfg)
        started = time.monotonic()
        if stop.is_set():
            return 0, True
        if payload["kind"] == "image":
            with Image.open(payload["source"]) as source:
                suffix = ".jpg" if source.format == "JPEG" else ".png"
            result = directory / ("result" + suffix)
            temporary = directory / ("result" + suffix + ".part")
            process_image(payload["source"], temporary, detector, cfg["max_image_pixels"],
                          redaction_mode=cfg.get("redaction_mode", "solid"))
            if stop.is_set():
                return 0, True
            temporary.replace(result)
            emit({"result_ref": str(result), "stats": {"processed_frames": 1, "dropped_frames": 0,
                                                       "elapsed_seconds": time.monotonic() - started}})
        elif payload["kind"] == "video":
            sampler = None
            if payload.get('profile_detection') and collector is None:
                from vision_shield.detection.device_profile import DeviceSamples

                primary = getattr(detector, 'primary', detector)
                if getattr(getattr(primary, 'device', None), 'type', None) != 'cuda':
                    raise ValueError('Detection profiling requires CUDA')
                sampler = primary.device_samples = DeviceSamples(directory, emit)
            temporary = directory / "result.mp4.part"
            process_video(payload["source"], str(temporary), detector, cfg, stop, emit)
            if not stop.is_set():
                if collector:
                    state = 'partial' if collector.device_stage and len(collector.device_samples) != len(collector.checkpoints) else 'complete'
                    report = dict(state=state, mode=profile_mode, model_load_seconds=load_seconds,
                                  runtime=metadata, profile=collector.report(collector.frames))
                    (directory / 'host-profile/report.json').write_text(json.dumps(report), 'utf-8')
                    emit({'device_profile': report})
                if sampler is not None:
                    sampler.finish()
                result = directory / "result.mp4"
                temporary.replace(result)
                emit({"result_ref": str(result)})
        elif payload["kind"] == "stream":
            process_stream(payload["source"], payload["publish_url"], detector, cfg, stop, emit)
            emit({"reconnect": True})
        else:
            raise ValueError("Unsupported task kind")
        return 0, True
    except Exception as error:
        if collector:
            emit({'device_profile': dict(state='failed', mode=profile_mode, reason=type(error).__name__)})
        if payload["kind"] == "probe":
            from vision_shield.detection.preflight import PreflightError
            emit({"preflight_error": str(error) if isinstance(error, PreflightError) else "模型预检失败：内部异常"})
            return 1, False
        model_failed = model_runtime_failure(error)
        retryable = payload["kind"] == "stream" and (isinstance(error, RETRYABLE_STREAM_ERRORS) or model_failed)
        code = "storage_full" if isinstance(error, OSError) and error.errno == 28 else "processing_failed"
        emit({"retryable": retryable, "failure_type": type(error).__name__,
              **({"failure_errno": error.errno} if isinstance(getattr(error, "errno", None), int) else {}),
              "failure_at": [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)],
              "error": {"code": code, "message": code.replace("_", " ")}})
        from vision_shield.media.streams import StreamCleanupError
        return 1, retryable and not model_failed and not isinstance(error, StreamCleanupError)
    finally:
        observers.close()


def main(*, persistent=False):
    payload = json.loads(sys.stdin.readline())
    stop = threading.Event()
    pending = Queue(maxsize=1)

    def watch_parent():
        nonlocal stop
        for line in sys.stdin:
            message = json.loads(line)
            if message.get('stop'):
                stop.set()
            else:
                stop = threading.Event()
                pending.put((message, stop))
        # EOF revokes an active or idle resident, including its CUDA context.
        os._exit(3)

    watcher = threading.Thread(target=watch_parent, daemon=True)
    if os.name != 'nt':
        watcher.start()
    if not persistent:
        # Windows CRT stdin reads must start after numerical DLL/model initialization.
        code, _ = process(payload, stop, on_initialized=watcher.start if os.name == 'nt' else None)
        if code:
            raise SystemExit(code)
        return

    try:
        from vision_shield.detection import load_detector
        from vision_shield.detection.preflight import verify_models, validate_device, run_inference
        # Load media dependencies before starting the Windows CRT control reader.
        from vision_shield.media import images, video, streams
        from PIL import Image

        cfg = payload['config']
        models = verify_models(cfg)
        validate_device(cfg.get('inference_device', 'cpu'))
        loading = time.monotonic()
        detector = load_detector(cfg)
        Image.init()
        load_seconds = time.monotonic() - loading
        warming = time.monotonic()
        report = {**run_inference(cfg, detector), 'models': models,
                  'model_load_seconds': load_seconds, 'warmup_seconds': time.monotonic() - warming}
    except Exception as error:
        from vision_shield.detection.preflight import PreflightError
        emit(dict(preflight_error=str(error) if isinstance(error, PreflightError) else '模型预检失败：内部异常',
                  retryable=model_runtime_failure(error), failure_type=type(error).__name__,
                  failure_at=failure_details(error, 'model_initialize')['at'],
                  error=dict(code='processing_failed', message='Model preflight failed')))
        emit(dict(complete=True, returncode=1, reusable=False))
        return
    if os.name == 'nt':
        watcher.start()
    emit(dict(probe=report))
    emit(dict(complete=True, returncode=0, reusable=True))
    while True:
        payload, task_stop = pending.get()
        code, reusable = process(payload, task_stop, detector)
        emit(dict(complete=True, returncode=code, reusable=reusable))
        if not reusable:
            return


if __name__ == "__main__":
    main(persistent='--persistent' in sys.argv)
