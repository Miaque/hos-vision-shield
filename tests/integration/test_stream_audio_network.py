"""隔离 RTSP 服务验证实际协商音轨，不依赖设备或部署环境。"""

import json
import os
import subprocess
import threading
import time
from itertools import pairwise
from pathlib import Path

import av

from tests.conftest import NO_WINDOW, ROOT, stop_process, unused_port, wait_for_port
from vision_shield.config import Settings
from vision_shield.media import streams


def test_native_rtsp_input_retains_audible_audio(generated_media, tmp_path, monkeypatch):
    """真实 RTSP 读取 H.264/AAC，默认保留音轨并正常输出脱敏画面。"""
    binary = Path(os.getenv("TEST_MEDIAMTX_BIN") or ROOT / ".tools/mediamtx/mediamtx.exe")
    assert binary.is_file(), "MediaMTX is required for the native RTSP input check"
    port = unused_port()
    address = f"rtsp://127.0.0.1:{port}/source"
    config_path = tmp_path / "mediamtx.yml"
    config_path.write_text(json.dumps({
        "logLevel": "warn", "rtsp": True, "rtspAddress": f"127.0.0.1:{port}",
        "rtspTransports": ["tcp"], "rtmp": False, "hls": False,
        "webrtc": False, "srt": False, "moq": False,
        "paths": {"source": {"source": "publisher"}},
    }), encoding="utf-8")
    target = tmp_path / "received.nut"
    processes = []
    with (tmp_path / "media.log").open("wb") as media_log, (tmp_path / "source.log").open("wb") as source_log:
        try:
            media = subprocess.Popen([str(binary), str(config_path)], stdout=media_log, stderr=media_log,
                                     cwd=tmp_path, creationflags=NO_WINDOW)
            processes.append(media)
            wait_for_port(media, port)
            publisher = subprocess.Popen(["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1",
                "-i", str(generated_media / "aac.mp4"), "-map", "0", "-c", "copy",
                "-f", "rtsp", "-rtsp_transport", "tcp", address], stdout=source_log, stderr=source_log,
                creationflags=NO_WINDOW)
            processes.append(publisher)
            original_open = av.open
            deadline = time.monotonic() + 10
            while True:
                assert publisher.poll() is None, source_log.name
                try:
                    with original_open(address, options={"rtsp_transport": "tcp"}, timeout=(1, 1)) as reader:
                        assert reader.streams.video and reader.streams.audio
                    break
                except av.error.FFmpegError:
                    assert time.monotonic() < deadline, "RTSP source did not become ready"
                    time.sleep(.05)
            monkeypatch.setattr(streams.av, "open", lambda url, *args, **kwargs:
                original_open(url, *args, **kwargs) if url == address else original_open(str(target), "w", format="nut"))

            class Detector:
                def detect(self, pixels):
                    return [(0, 0, 320, 240)]

            stop = threading.Event()
            config = Settings(_env_file=None, rtsp_connect_timeout=5, rtsp_read_timeout=5).model_dump()
            streams.process_stream(address, "rtsp://output/live", Detector(), config, stop,
                lambda event: stop.set() if event.get("stats", {}).get("processed_frames", 0) >= 24 else None)
        finally:
            for process in reversed(processes):
                stop_process(process)
    with original_open(str(target)) as reader:
        assert reader.streams.audio[0].codec_context.name == "aac"
        video = list(reader.decode(video=0))
    with original_open(str(target)) as reader:
        audio = list(reader.decode(audio=0))
    assert len(video) >= 24 and audio
    assert video[-1].to_ndarray(format="rgb24").mean() < 100
    assert all(b.pts * b.time_base > a.pts * a.time_base for a, b in pairwise(video))
    assert all(b.pts * b.time_base > a.pts * a.time_base for a, b in pairwise(audio))
    assert abs(audio[-1].pts * audio[-1].time_base - video[-1].pts * video[-1].time_base) < .2
