"""真实编解码验证；只将网络端点替换为临时媒体文件。"""

import threading
import time
import os
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.detection.timing import StageTimes
from vision_shield.media import streams
from vision_shield.media.stream_metrics import vcl_digest
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
        stage_times = StageTimes()

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
    assert events[-1]['timings']['source-read'] >= 0.04
    assert events[-1]['timings']['source-decode'] > 0
    assert events[-1]['timings']['source-backpressure'] > 0
    assert [sample['sequence'] for sample in metrics['recent_outputs']] == list(range(1, 13))
    assert all(sample['latency_seconds'] >= 0 for sample in metrics['recent_outputs'])
    assert opened["format"] == ("flv" if scheme == "rtmp" else "rtsp")
    if scheme == "rtmp":
        assert "rtsp_transport" not in opened["options"]
        assert opened["options"]["rtmp_tcurl"] == "rtmp://media.test/live"
    else:
        assert opened["options"]["rtsp_transport"] == "tcp"
    with original_open(str(target)) as reader:
        packets = [p for p in reader.demux(video=0) if p.size]
        if scheme == 'rtmp':
            convert = av.bitstream.BitStreamFilterContext('h264_mp4toannexb', reader.streams.video[0])
            packets = [output for packet in packets for output in convert.filter(packet)]
        assert len(packets) == 12 and 0 < sum(p.is_keyframe for p in packets) < 12
        assert [s['encoded_vcl_sha256'] for s in metrics['recent_outputs']] == [vcl_digest(bytes(p)) for p in packets]
        assert all(s['encoded_bytes'] > 0 and s['encoded_vcl_sha256'] for s in metrics['recent_outputs'])
    with original_open(str(target)) as reader:
        assert reader.streams.video[0].codec_context.name == "h264"
        output_audio = reader.streams.audio
        assert bool(output_audio) is bool(audio_codec)
        if audio_codec:
            assert output_audio[0].codec_context.name == ("aac" if scheme == "rtmp" else audio_codec)
        frames = list(reader.decode(video=0))
        assert len(frames) == 12 and frames[-1].to_ndarray(format="rgb24").mean() < 100
        pixels = frames[-1].to_ndarray(format="rgb24")
        np.testing.assert_allclose(pixels[24:].mean(axis=(0, 1)), [64, 64, 64], atol=5)
        assert any(event.get("output") for event in events)
    if audio_codec:
        with original_open(str(target)) as reader:
            assert sum(frame.samples for frame in reader.decode(audio=0)) > 0


@pytest.mark.parametrize('clocks,duration,expected', [
    ([10000, 11000, 11000, 9000, 10000], 0, [10000, 11000, 12000, 13000, 14000]),
    ([10000, 10000, 11000], 1000, [10000, 11000, 12000]),
    ([10000, 10000], 0, None),
])
def test_realtime_clock_repair_keeps_independent_pictures_and_raw_clocks(tmp_path, monkeypatch, clocks, duration, expected):
    target = tmp_path / 'repaired.nut'
    source_video = SimpleNamespace(type='video', width=64, height=48, average_rate=None,
                                   codec_context=SimpleNamespace(name='h264'))
    frames = []
    for index, pts in enumerate(clocks):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 80 + index * 25, dtype=np.uint8), format='rgb24')
        frame.pts, frame.time_base, frame.duration = pts, Fraction(1, 12000), duration
        frames.append(frame)

    class Reader:
        streams = SimpleNamespace(video=[source_video], audio=[])

        def demux(self):
            for frame in frames:
                yield SimpleNamespace(stream=source_video, pts=frame.pts, time_base=frame.time_base,
                                      decode=lambda frame=frame: [frame])

        def close(self):
            pass

    original_open = av.open
    monkeypatch.setattr(streams.av, 'open', lambda url, *args, **kwargs: Reader()
                        if url.startswith('rtsp://camera/') else original_open(str(target), 'w', format='nut'))

    class Detector:
        def detect(self, pixels):
            return [(0, 0, 32, 48)]

    events = []
    config = Settings(_env_file=None).model_dump()
    config['inference_device'] = 'cpu'
    def run():
        streams.process_stream('rtsp://camera/source', 'rtsp://output/test', Detector(), config,
                               threading.Event(), events.append)

    if expected is None:
        with pytest.raises(BufferError, match='frame interval'):
            run()
        assert events[-1]['stats']['stream']['encoded_frames'] == 1
        assert events[-1]['stats']['stream']['failed_frames'] == 1
        return
    run()
    stats = events[-1]['stats']
    metrics = stats['stream']
    assert stats['dropped_frames'] == metrics['failed_frames'] == metrics['unmatched_packets'] == 0
    assert metrics['input_frames'] == metrics['encoded_frames'] == len(clocks)
    assert [s['source_pts'] for s in metrics['recent_outputs']] == [str(Fraction(p, 12000)) for p in clocks]
    assert [s['timeline_pts'] for s in metrics['recent_outputs']] == [str(Fraction(p, 12000)) for p in expected]
    assert metrics['timestamp_repairs'] == (2 if len(clocks) == 5 else 1)
    with original_open(str(target)) as reader:
        output = list(reader.decode(video=0))
    assert [f.pts * f.time_base for f in output] == [Fraction(p - expected[0], 12000) for p in expected]
    for index, frame in enumerate(output):
        pixels = frame.to_ndarray(format='rgb24')
        assert pixels[:, :24].mean() < 70
        assert abs(pixels[:, 40:].mean() - (80 + index * 25)) < 5
