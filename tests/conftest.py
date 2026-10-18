"""Real service fixtures use isolated ports and never touch existing databases."""

import json
import os
import shutil
import socket
import subprocess
import time
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_port(process, port: int) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Test service exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("Test service did not start")


def stop_process(process) -> None:
    if os.name == "nt" and process.poll() is None:
        # venv python.exe is a launcher; terminate its owned interpreter tree too.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=NO_WINDOW, timeout=10)
        process.wait(timeout=5)
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.fixture(scope="session")
def sample_manifest():
    return json.loads((ROOT / "tests/fixtures/manifest.json").read_text("utf-8"))


@pytest.fixture(scope="session")
def redis_url(tmp_path_factory):
    binary = os.getenv("TEST_REDIS_SERVER") or shutil.which("redis-server")
    if not binary or not Path(binary).is_file():
        pytest.fail("Set TEST_REDIS_SERVER to a Redis 7+ executable; no mock fallback")
    directory = tmp_path_factory.mktemp("redis")
    port = unused_port()
    with (directory / "redis.log").open("wb") as log:
        process = subprocess.Popen(
            [binary, "--bind", "127.0.0.1", "--port", str(port), "--save", "",
             "--appendonly", "no", "--dir", "."],
            cwd=directory, stdout=log, stderr=log, creationflags=NO_WINDOW,
        )
        try:
            wait_for_port(process, port)
            yield f"redis://127.0.0.1:{port}/0"
        finally:
            stop_process(process)


@pytest.fixture(scope="session")
def generated_media(tmp_path_factory):
    from PIL import Image

    binary = os.getenv("TEST_FFMPEG") or shutil.which("ffmpeg")
    if not binary:
        pytest.fail("FFmpeg is required to generate real media fixtures")
    directory = tmp_path_factory.mktemp("media")
    for fmt in ("PNG", "JPEG"):
        Image.new("RGB", (320, 240), (110, 160, 210)).save(directory / f"no-target.{fmt.lower()}")
    for audio in (False, True):
        command = [binary, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12:duration=2"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "2"]
        if audio:
            command += ["-c:a", "aac"]
        command += [str(directory / ("aac.mp4" if audio else "silent.mp4"))]
        subprocess.run(command, check=True, capture_output=True, timeout=30, creationflags=NO_WINDOW)
    subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=12:duration=2", "-vf",
         "setpts=if(lt(N\\,12)\\,N/(12*TB)\\,(N-12)/(6*TB)+1/TB)",
         "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(directory / "vfr.mp4")],
        check=True, capture_output=True, timeout=30, creationflags=NO_WINDOW,
    )
    return directory


@pytest.fixture
def http_media(generated_media):
    handler = partial(SimpleHTTPRequestHandler, directory=str(generated_media))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", generated_media
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
