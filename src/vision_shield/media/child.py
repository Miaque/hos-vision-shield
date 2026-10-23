"""One media attempt. EOF on the private control pipe revokes this process."""

import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from contextlib import ExitStack

from av.error import FFmpegError

# PyAV ArgumentError is a ValueError; a disrupted mux still needs a fresh session.
RETRYABLE_STREAM_ERRORS = (OSError, BufferError, FFmpegError)


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    payload = json.loads(sys.stdin.readline())
    stop = threading.Event()
    observers = ExitStack()
    collector = None

    def watch_parent():
        for line in sys.stdin:
            if json.loads(line).get("stop"):
                stop.set()
        # The supervisor died; no detached process may keep a camera/publisher alive.
        os._exit(3)

    # Linux 在冷加载前监听父管道，防止父进程退出后遗留加载中的模型进程。
    if os.name != "nt":
        threading.Thread(target=watch_parent, daemon=True).start()

    try:
        cfg = payload["config"]
        if payload["kind"] == "probe":
            from vision_shield.detection.preflight import run_preflight
            report = run_preflight(cfg)
            if os.name == "nt":
                threading.Thread(target=watch_parent, daemon=True).start()
            if not stop.is_set():
                emit({"probe": report})
            return
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
                    or cfg['detection_backend'] != 'quality'):
                raise ValueError('Model profiling requires quality/CUDA')
            collector = ProfileTimes(directory / 'host-profile',
                                     device_stage=None if profile_mode == 'host' else profile_mode)
            observers.callback(collector.close)
            instrument(observers, collector)
        loading = time.monotonic()
        detector = load_detector(cfg)
        Image.init()
        load_seconds = time.monotonic() - loading
        emit({"model_load_seconds": load_seconds})
        if collector:
            from vision_shield.detection.host_profile import instrument_candidates, runtime_metadata

            instrument_candidates(detector.primary, 'primary', collector)
            instrument_candidates(detector.detail, 'detail', collector)
            metadata = runtime_metadata(detector, cfg)
        # Windows numerical DLL initialization can deadlock against a blocking
        # CRT stdin reader. Load dependencies before starting the control thread.
        if os.name == "nt":
            threading.Thread(target=watch_parent, daemon=True).start()
        started = time.monotonic()
        if stop.is_set():
            return
        if payload["kind"] == "image":
            with Image.open(payload["source"]) as source:
                suffix = ".jpg" if source.format == "JPEG" else ".png"
            result = directory / ("result" + suffix)
            temporary = directory / ("result" + suffix + ".part")
            process_image(payload["source"], temporary, detector, cfg["max_image_pixels"],
                          redaction_mode=cfg.get("redaction_mode", "solid"))
            if stop.is_set():
                return
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
    except Exception as error:
        if collector:
            emit({'device_profile': dict(state='failed', mode=profile_mode, reason=type(error).__name__)})
        if payload["kind"] == "probe":
            from vision_shield.detection.preflight import PreflightError
            emit({"preflight_error": str(error) if isinstance(error, PreflightError) else "模型预检失败：内部异常"})
            sys.exit(1)
        retryable = payload["kind"] == "stream" and isinstance(error, RETRYABLE_STREAM_ERRORS)
        code = "storage_full" if isinstance(error, OSError) and error.errno == 28 else "processing_failed"
        emit({"retryable": retryable, "failure_type": type(error).__name__,
              "failure_at": [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)],
              "error": {"code": code, "message": code.replace("_", " ")}})
        sys.exit(1)
    finally:
        observers.close()


if __name__ == "__main__":
    main()
