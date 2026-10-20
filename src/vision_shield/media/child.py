"""One media attempt. EOF on the private control pipe revokes this process."""

import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

from av.error import FFmpegError

# PyAV ArgumentError is a ValueError; a disrupted mux still needs a fresh session.
RETRYABLE_STREAM_ERRORS = (OSError, BufferError, FFmpegError)


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    payload = json.loads(sys.stdin.readline())
    stop = threading.Event()

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

        loading = time.monotonic()
        detector = load_detector(cfg)
        Image.init()
        emit({"model_load_seconds": time.monotonic() - loading})
        # Windows numerical DLL initialization can deadlock against a blocking
        # CRT stdin reader. Load dependencies before starting the control thread.
        if os.name == "nt":
            threading.Thread(target=watch_parent, daemon=True).start()
        directory = Path(payload["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        if stop.is_set():
            return
        if payload["kind"] == "image":
            with Image.open(payload["source"]) as source:
                suffix = ".jpg" if source.format == "JPEG" else ".png"
            result = directory / ("result" + suffix)
            temporary = directory / ("result" + suffix + ".part")
            process_image(payload["source"], temporary, detector, cfg["max_image_pixels"])
            if stop.is_set():
                return
            temporary.replace(result)
            emit({"result_ref": str(result), "stats": {"processed_frames": 1, "dropped_frames": 0,
                                                       "elapsed_seconds": time.monotonic() - started}})
        elif payload["kind"] == "video":
            temporary = directory / "result.mp4.part"
            process_video(payload["source"], str(temporary), detector, cfg, stop, emit)
            if not stop.is_set():
                result = directory / "result.mp4"
                temporary.replace(result)
                emit({"result_ref": str(result)})
        elif payload["kind"] == "stream":
            process_stream(payload["source"], payload["publish_url"], detector, cfg, stop, emit)
            emit({"reconnect": True})
        else:
            raise ValueError("Unsupported task kind")
    except Exception as error:
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


if __name__ == "__main__":
    main()
