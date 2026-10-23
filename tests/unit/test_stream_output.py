"""真实编解码验证；只将网络端点替换为临时媒体文件。"""

import threading
import time
import os

import av
import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.media import streams
from vision_shield.media.video import check_streams


@pytest.mark.parametrize("scheme", ["rtsp", "rtmp"])
@pytest.mark.parametrize("audio_codec", [None, "aac", "pcm_alaw", "pcm_mulaw"])
@pytest.mark.parametrize("fault", [None, TimeoutError, RuntimeError])
@pytest.mark.parametrize("source_bframes", [0, 2])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.getenv("RUN_GPU_TESTS") != "1", reason="requires real NVDEC/GPU mask/NVENC"))])
def test_stream_output_muxes_processed_video_and_compatible_audio(tmp_path, monkeypatch, scheme, audio_codec, fault, source_bframes, device):
    source, target = tmp_path / "source.nut", tmp_path / "output.nut"
    with av.open(str(source), "w") as writer:
        video = writer.add_stream("libx264", rate=12)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        video.options = {"tune": "zerolatency", "bf": str(source_bframes)}
        audio = writer.add_stream(audio_codec, rate=8000 if audio_codec != "aac" else 48000) if audio_codec else None
        if audio:
            audio.layout = "mono"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 180, dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame):
                writer.mux(packet)
            if audio:
                frame = av.AudioFrame(format="s16", layout="mono", samples=audio.rate // 12)
                frame.sample_rate = audio.rate
                for plane in frame.planes:
                    plane.update(bytes(plane.buffer_size))
                for packet in audio.encode(frame):
                    writer.mux(packet)
        for stream in (video, audio):
            if stream:
                for packet in stream.encode():
                    writer.mux(packet)

    original_open = av.open
    opened = {}
    if audio_codec in {"pcm_alaw", "pcm_mulaw"}:
        with original_open(str(source)) as reader:
            with pytest.raises(ValueError, match="audio codec"):
                check_streams(reader, Settings(_env_file=None).model_dump())

    class Reader:
        def __init__(self, hardware):
            self.container = original_open(str(source), **hardware)
            self.streams = self.container.streams

        def demux(self):
            for packet in self.container.demux():
                time.sleep(0.005)
                yield packet

        def close(self):
            self.container.close()

    def open_endpoint(url, *args, **kwargs):
        if url.startswith("rtsp://camera/"):
            return Reader({key: kwargs[key] for key in ("hwaccel",) if key in kwargs})
        opened.update(url=url, **kwargs)
        return original_open(str(target), "w", format="flv" if scheme == "rtmp" else "nut")

    class Detector:
        def detect(self, pixels):
            if fault:
                raise fault("inference failed")
            return [(0, 0, 64, 48)]

    monkeypatch.setattr(streams.av, "open", open_endpoint)
    events = []
    config = Settings(_env_file=None).model_dump(mode="json")
    config["inference_device"] = device
    config["publish_url"] = f"{scheme}://media.test/live/camera?sign=secret"
    if fault:
        with pytest.raises(fault, match="inference failed"):
            streams.process_stream("rtsp://camera/source", f"{scheme}://127.0.0.1/live/camera?sign=secret",
                                   Detector(), config, threading.Event(), events.append)
        assert not any(event.get("output") for event in events)
        assert not target.exists() or target.stat().st_size == 0
        assert events[-1]['stats']['stream']['encoded_frames'] == 0
        assert events[-1]['stats']['stream']['failed_frames'] == 1
        return
    streams.process_stream("rtsp://camera/source", f"{scheme}://127.0.0.1/live/camera?sign=secret",
                           Detector(), config, threading.Event(), events.append)
    metrics = events[-1]['stats']['stream']
    assert metrics['input_frames'] == metrics['encoded_frames'] == 12
    assert metrics['unmatched_packets'] == metrics['missing_receive_frames'] == metrics['failed_frames'] == 0
    assert metrics['protected_frames'] == metrics['pending_encoder_frames'] == 0
    assert [sample['sequence'] for sample in metrics['recent_outputs']] == list(range(1, 13))
    assert all(sample['latency_seconds'] >= 0 for sample in metrics['recent_outputs'])
    assert opened["format"] == ("flv" if scheme == "rtmp" else "rtsp")
    if scheme == "rtmp":
        assert "rtsp_transport" not in opened["options"]
        assert opened["options"]["rtmp_tcurl"] == "rtmp://media.test/live"
    else:
        assert opened["options"]["rtsp_transport"] == "tcp"
    with original_open(str(target)) as reader:
        assert reader.streams.video[0].codec_context.name == "h264"
        output_audio = reader.streams.audio
        assert bool(output_audio) is bool(audio_codec)
        if audio_codec:
            assert output_audio[0].codec_context.name == ("aac" if scheme == "rtmp" else audio_codec)
        frames = list(reader.decode(video=0))
        assert len(frames) == 12 and frames[-1].to_ndarray(format="rgb24").mean() < 100
        assert any(event.get("output") for event in events)
    if audio_codec:
        with original_open(str(target)) as reader:
            assert sum(frame.samples for frame in reader.decode(audio=0)) > 0
