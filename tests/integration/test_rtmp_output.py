"""使用隔离 FFmpeg 接收端验证 RTMP 握手、传输和可解码输出。"""

import socket
import subprocess
import threading
import time

import av
import pytest

from tests.conftest import NO_WINDOW, stop_process, unused_port
from vision_shield.config import Settings
from vision_shield.media.streams import process_stream


@pytest.mark.parametrize("audio_codec", [None, "aac", "pcm_alaw"])
def test_rtmp_receiver_gets_redacted_h264_and_aac(generated_media, tmp_path, audio_codec):
    source = generated_media / ("aac.mp4" if audio_codec else "silent.mp4")
    if audio_codec == "pcm_alaw":
        converted = tmp_path / "g711.nut"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:v", "copy", "-c:a", "pcm_alaw",
                        "-ar", "8000", str(converted)], check=True, capture_output=True, timeout=15, creationflags=NO_WINDOW)
        source = converted
    port = unused_port()
    address = f"rtmp://127.0.0.1:{port}/app/camera?sign=test"
    result = tmp_path / "received.flv"

    class Detector:
        def detect(self, pixels):
            return [(0, 0, 320, 240)]

    with (tmp_path / "receiver.log").open("wb") as log:
        receiver = subprocess.Popen(["ffmpeg", "-v", "error", "-rw_timeout", "5000000", "-listen", "1", "-timeout", "5",
            "-i", address, "-map", "0", "-c", "copy", str(result)], stdout=log, stderr=log, creationflags=NO_WINDOW)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                assert receiver.poll() is None, "RTMP receiver exited before listening"
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError:
                        break
                time.sleep(0.02)
            else:
                pytest.fail("RTMP receiver did not start")
            config = Settings(_env_file=None, rtsp_connect_timeout=5, rtsp_read_timeout=5).model_dump(mode="json")
            process_stream(str(source), address, Detector(), config, threading.Event(), lambda event: None)
            assert receiver.wait(timeout=10) == 0
        finally:
            stop_process(receiver)

    with av.open(str(result)) as reader:
        assert reader.streams.video[0].codec_context.name == "h264"
        assert bool(reader.streams.audio) is bool(audio_codec)
        if audio_codec:
            assert reader.streams.audio[0].codec_context.name == "aac"
        frames = list(reader.decode(video=0))
        assert frames and frames[-1].to_ndarray(format="rgb24").mean() < 100
    if audio_codec:
        with av.open(str(result)) as reader:
            assert sum(frame.samples for frame in reader.decode(audio=0)) > 0
