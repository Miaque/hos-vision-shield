"""真实音视频编解码回归：异常输入时钟不能丢声音、画面或累积音画偏移。"""

import threading
from fractions import Fraction
from itertools import pairwise
from types import SimpleNamespace

import av
import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.media import streams


@pytest.mark.parametrize("protocol", ["rtsp", "rtmp"])
@pytest.mark.parametrize("audio_codec", ["aac", "pcm_alaw", "pcm_mulaw"])
@pytest.mark.parametrize("fault", ["none", "video_repeat", "video_reset", "audio_repeat", "audio_reset", "both_reset"])
def test_live_audio_clock_recovery_preserves_audio_and_sync(
    tmp_path, monkeypatch, protocol, audio_codec, fault, *, coarse_clock=False, source_bframes=0,
    debug=True, e0=True, capture=None,
):
    """反复重复 PTS、单轨与双轨回退后，独立画面及 AAC 内容仍保持同一时间线。"""
    source, target = tmp_path / "source.nut", tmp_path / "output.nut"
    fps, count, rate = 12, 120, 48000 if audio_codec == "aac" else 8000
    with av.open(str(source), "w") as writer:
        video = writer.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        video.options = {"tune": "zerolatency", "bf": str(source_bframes)}
        audio = writer.add_stream(audio_codec, rate=rate)
        audio.layout = "mono"
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 90 + index % 8 * 15, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in video.encode(frame):
                writer.mux(packet)
            samples = np.arange(rate // fps) + index * (rate // fps)
            wave = (np.sin(samples * (2 * np.pi * 440 / rate)) * 12000).astype(np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(wave, format="s16", layout="mono")
            frame.pts, frame.time_base, frame.sample_rate = index * (rate // fps), Fraction(1, rate), rate
            for packet in audio.encode(frame):
                writer.mux(packet)
        for track in (video, audio):
            for packet in track.encode():
                writer.mux(packet)

    original_open = av.open
    with original_open(str(source)) as reader:
        original_audio = [(bytes(p), p.pts * p.time_base) for p in reader.demux(audio=0) if p.size]
    with original_open(str(source)) as reader:
        original_video = [f.pts * f.time_base for f in reader.decode(video=0)]
    raw_clocks = []

    class Reader:
        def __init__(self):
            self.container = original_open(str(source))
            self.streams = self.container.streams

        def demux(self, **kwargs):
            video_index = audio_index = 0
            previous_audio = None
            for packet in self.container.demux(**kwargs):
                if packet.stream.type == "video":
                    decoded = packet.decode()
                    for frame in decoded:
                        if coarse_clock:
                            base = frame.time_base
                            frame.pts = round(frame.pts * base * 1000)
                            frame.duration = round(frame.duration * base * 1000)
                            frame.time_base = Fraction(1, 1000)
                        if fault == "video_repeat" and video_index >= 18 and video_index % 9 == 0:
                            frame.pts -= round(Fraction(1, fps) / frame.time_base)
                        if fault in {"video_reset", "both_reset"} and video_index >= 48:
                            frame.pts -= round(Fraction(2) / frame.time_base)
                        raw_clocks.append(frame.pts * frame.time_base)
                        video_index += 1
                        yield SimpleNamespace(stream=packet.stream, pts=frame.pts, time_base=frame.time_base,
                                              decode=lambda frame=frame: [frame])
                else:
                    if coarse_clock and packet.dts is not None:
                        base = packet.time_base
                        packet.pts = round(packet.pts * base * 1000)
                        packet.dts = round(packet.dts * base * 1000)
                        packet.duration = round(packet.duration * base * 1000)
                        packet.time_base = Fraction(1, 1000)
                    if packet.dts is not None:
                        raw = packet.dts
                        if fault == "audio_repeat" and audio_index >= 46 and audio_index % 23 == 0:
                            packet.pts += previous_audio - raw
                            packet.dts = previous_audio
                        if fault in {"audio_reset", "both_reset"} and raw * packet.time_base >= 4:
                            shift = round(Fraction(2) / packet.time_base)
                            packet.pts -= shift
                            packet.dts -= shift
                        previous_audio = raw
                        audio_index += 1
                    yield packet

        def close(self):
            self.container.close()

    monkeypatch.setattr(streams.av, "open", lambda url, *args, **kwargs: Reader()
        if url == "rtmp://camera/input" else original_open(str(target), "w", format="flv" if protocol == "rtmp" else "nut"))

    class Detector:
        def detect(self, pixels):
            return [(0, 0, 32, 48)]

    events = []
    config = Settings(_env_file=None, debug=debug).model_dump()
    config.update(inference_device="cpu", audio_enabled=True, stream_e0_diagnostics=e0)
    streams.process_stream("rtmp://camera/input", f"{protocol}://output/live", Detector(), config,
                           threading.Event(), events.append)
    assert events[-1]["stats"]["processed_frames"] == count
    if debug:
        metrics = events[-1]["stats"]["stream"]
        assert metrics["encoded_frames"] == count and metrics["failed_frames"] == 0
        assert [s["source_pts"] for s in metrics["recent_outputs"]] == [str(value) for value in raw_clocks]
        if fault in {"video_repeat", "video_reset", "both_reset"}:
            assert metrics["timestamp_repairs"] > 0
    else:
        assert all("stream" not in event.get("stats", {}) for event in events)
    with original_open(str(target)) as reader:
        output_audio = [(bytes(p), p.pts * p.time_base) for p in reader.demux(audio=0) if p.size]
    assert all(b[1] > a[1] for a, b in pairwise(output_audio))
    transcoded = protocol == "rtmp" and audio_codec != "aac"
    if not transcoded:
        assert len(original_audio) - 6 <= len(output_audio) <= len(original_audio)
        assert [p[0] for p in output_audio] == [p[0] for p in original_audio[:len(output_audio)]]
    with original_open(str(target)) as reader:
        output_video = list(reader.decode(video=0))
    assert len(output_video) == count
    first_video = output_video[0].pts * output_video[0].time_base
    for index, frame in enumerate(output_video):
        assert abs((frame.pts * frame.time_base - first_video) - (original_video[index] - original_video[0])) < Fraction(2, 1000)
        pixels = frame.to_ndarray(format="rgb24")
        assert pixels[:, :24].mean() < 70
        assert abs(pixels[:, 40:].mean() - (90 + index % 8 * 15)) < 5
    if not transcoded:
        for index, (_, moment) in enumerate(output_audio):
            assert abs((moment - first_video) - (original_audio[index][1] - original_video[0])) < Fraction(2, 1000)
    else:
        with original_open(str(target)) as reader:
            decoded = list(reader.decode(audio=0))
        assert all(frame.sample_rate == 48000 for frame in decoded)
        assert count / fps - .15 < sum(frame.samples for frame in decoded) / 48000 < count / fps + .05
        assert abs(output_audio[0][1] - first_video) < Fraction(1, fps)
        assert abs(output_audio[-1][1] - output_video[-1].pts * output_video[-1].time_base) < Fraction(1, fps)
        wave = np.concatenate([frame.to_ndarray().reshape(-1) for frame in decoded])[48000:96000]
        assert np.sqrt(np.mean(wave ** 2)) > .01
        assert abs(np.argmax(np.abs(np.fft.rfft(wave))) - 440) <= 2
    if capture is not None:
        capture.update(target=target, events=events)


def test_unknown_audio_interval_still_fails_closed():
    """无可靠音频间隔不能凭空生成时钟；缺失时间戳也不能进入配对队列。"""
    clock = streams.LiveAVClock()
    packet = av.Packet(b"audio")
    packet.pts = packet.dts = 0
    packet.time_base = Fraction(1, 48000)
    clock.audio_packet(packet)
    with pytest.raises(BufferError, match="packet interval"):
        clock.audio_packet(packet)
    packet.pts = None
    with pytest.raises(BufferError, match="Invalid audio clock"):
        streams.LiveAVClock().audio_packet(packet)


@pytest.mark.parametrize("protocol", ["rtsp", "rtmp"])
@pytest.mark.parametrize("fault", ["video_reset", "audio_reset", "both_reset"])
def test_millisecond_input_does_not_accumulate_rounding_drift(tmp_path, monkeypatch, protocol, fault):
    """RTMP 毫秒时钟下，音频采样与视频帧周期的小数部分必须跨包保留。"""
    test_live_audio_clock_recovery_preserves_audio_and_sync(
        tmp_path, monkeypatch, protocol, "aac", fault, coarse_clock=True,
    )


@pytest.mark.parametrize("fault", ["none", "video_repeat", "both_reset"])
def test_production_reordered_video_keeps_audio_sync(tmp_path, monkeypatch, fault):
    """有 B 帧且关闭调试时仍修复两轨时钟，音轨内容和画面均不丢失。"""
    test_live_audio_clock_recovery_preserves_audio_and_sync(
        tmp_path, monkeypatch, "rtmp", "aac", fault, source_bframes=2, debug=False,
    )


def test_recovery_jumps_do_not_become_a_new_frame_interval():
    """孤立坏时间戳后的追赶跨越两帧，不能将其记作单帧周期而逐渐漂移。"""
    clock = streams.LiveAVClock()
    actual = []
    for pts in [0, 1, 1, 3, 3, 5, 5, 7]:
        frame = SimpleNamespace(pts=pts, time_base=Fraction(1, 25), duration=0)
        clock.video_frame(frame)
        actual.append(frame.pts)
    assert actual == list(range(8))
    clock = streams.LiveAVClock()
    actual = []
    for pts in [0, 1024, 1024, 3072, 3072, 5120, 5120, 7168]:
        packet = av.Packet(b"audio")
        packet.pts = packet.dts = pts
        packet.time_base = Fraction(1, 48000)
        clock.audio_packet(packet)
        actual.append(packet.pts)
    assert actual == [index * 1024 for index in range(8)]
