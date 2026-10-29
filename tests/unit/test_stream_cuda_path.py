"""CPU contract checks; actual NVDEC/masking/NVENC still require the GPU server."""

from fractions import Fraction
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import av
import torch
import numpy as np

from vision_shield.config import Settings
from vision_shield.media import cuda_frames, streams
from vision_shield.media.plate_lookahead import PendingMask


@pytest.mark.parametrize("fault", [None, "decode", "mask", "encode", "mux"])
@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize('debug', [False, True])
def test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(monkeypatch, fault, delayed, repair, debug, muted=False, with_audio=False, default_schedule=False, pacing=True, slow_first=False):
    source_video = SimpleNamespace(width=64, height=48, codec_context=SimpleNamespace(name="h264"))
    frames = [SimpleNamespace(width=64, height=48, pts=900000 + 3600 * i, time_base=Fraction(1, 90000))
              for i in range(3)]
    if repair:
        frames[-1].pts = frames[-2].pts
    threads, masked, encoded, published_at = {}, [], [], []
    closed = threading.Event()
    accelerator = object()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: threads.update(selected=device))

    class Reader:
        streams = SimpleNamespace(video=[source_video], audio=[SimpleNamespace(
            codec_context=SimpleNamespace(name="aac"))] if muted or with_audio else [])

        def demux(self, **kwargs):
            for frame in frames:
                yield SimpleNamespace(stream=source_video, pts=frame.pts, time_base=frame.time_base,
                                      decode=lambda frame=frame: [frame])

        def close(self):
            closed.set()

    source_video.type = "video"

    class DeviceFrames:
        device = 2

        def __init__(self):
            assert threads["selected"] == 2
            threads["context"] = threading.get_ident()

        def masked(self, frame, redactor, *, defer=False):
            assert redactor.detection_interval == 3 and redactor.tracking == "flow"
            assert defer and redactor._plate_template_fallback
            if fault == "mask":
                raise RuntimeError("mask failed")
            masked.append(frame.pts)
            output = SimpleNamespace(**vars(frame))
            return PendingMask(float(frame.pts * frame.time_base), np.zeros((48, 64), np.uint8), [], [],
                               lambda extra: output)

    class Writer:
        def add_stream_from_template(self, source):
            return SimpleNamespace()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def mux(self, packet):
            if fault == "mux":
                raise RuntimeError("mux failed")
            encoded.append(packet.pts)
            published_at.append(time.monotonic())
            packet.pts = -1  # Muxers may rescale/mutate the packet after accepting it.

    def open_endpoint(url, *args, **kwargs):
        if url == ("rtmp://camera/input" if muted or with_audio else "rtsp://camera/input"):
            threads["decode"] = threading.get_ident()
            assert kwargs["hwaccel"] is accelerator
            if not muted and not with_audio:
                assert kwargs["options"]["allowed_media_types"] == "video+audio"
            if fault == "decode":
                raise RuntimeError("decode failed")
            return Reader()
        return Writer()

    pending = []

    def encode(frame=None):
        if frame is None:
            return pending
        if fault == "encode":
            raise RuntimeError("encode failed")
        if slow_first and frame.pts == 0:
            time.sleep(.08)
        packet = av.Packet(b'\x00\x00\x00\x01\x65picture')
        packet.pts, packet.time_base = frame.pts, frame.time_base
        if delayed:
            pending.append(packet)
            return []
        return [packet]

    output_video = SimpleNamespace(width=64, height=48, encode=encode,
                                   codec_context=SimpleNamespace(name="h264_nvenc"))
    add_video = Mock(return_value=output_video)
    software_mask = Mock(side_effect=AssertionError("CPU frame conversion must not run"))
    monkeypatch.setattr(cuda_frames, "CUDAFrames", DeviceFrames)
    monkeypatch.setattr(cuda_frames, "hwaccel", lambda device: accelerator if device == 2 else None)
    monkeypatch.setattr(streams.av, "open", open_endpoint)
    monkeypatch.setattr(streams, "add_video", add_video)
    monkeypatch.setattr(streams, "masked_frame", software_mask)
    paced = []
    wait = streams.StreamPacer.wait
    monkeypatch.setattr(streams.StreamPacer, 'wait', lambda self, moment: paced.append(moment) or wait(self, moment))
    config = {**Settings(_env_file=None, debug=debug).model_dump(), "inference_device": "cuda",
              "detection_interval": 3, "tracking": "flow", "stream_output_pacing": pacing}
    if default_schedule:
        config.pop('detection_interval')
        config.pop('tracking')
    if muted:
        config["audio_enabled"] = False
    events = []
    stop = threading.Event()

    def process():
        streams.process_stream("rtmp://camera/input" if muted or with_audio else "rtsp://camera/input",
                               "rtsp://output/test", object(), config, stop, events.append)

    if fault:
        with pytest.raises(RuntimeError, match=f"{fault} failed"):
            process()
        assert not encoded and not any(event.get("output") for event in events)
        if fault == "mux" and debug:
            assert events[-1]['stats']['stream']['failed_frames'] == 1
            assert events[-1]['stats']['stream']['encoded_frames'] == 0
    else:
        process()
        assert masked == [900000, 903600, 907200]
        assert encoded == [0, 3600, 7200]
        assert len(paced) == (3 if pacing else 0)
        if slow_first:
            assert all(b - a >= .018 for a, b in zip(published_at, published_at[1:]))
            assert events[-1]['timings']['output-pacing'] > .03
        assert threads["context"] == threads["decode"] != threading.get_ident()
        assert add_video.call_args.kwargs == {"live": True, "device": "cuda"}
        assert events[-1]["stats"]["processed_frames"] == 3
        if debug:
            assert events[-1]["stats"]["stream"]["encoded_frames"] == 3
            assert events[-1]['stats']['stream']['lookahead_frames'] == 0
            assert events[-1]['detection_stats']['lookahead_max_frames'] == 3
            assert (events[-1]['stats']['stream']['input_width'], events[-1]['stats']['stream']['input_height']) == (64, 48)
            assert events[-1]["stats"]["stream"]["missing_receive_frames"] == 0
            assert events[-1]["stats"]["stream"]["timestamp_repairs"] == int(repair)
            assert [s['source_pts'] for s in events[-1]['stats']['stream']['recent_outputs']] == [
                '10', '251/25', '251/25' if repair else '252/25']
            for sample in events[-1]['stats']['stream']['recent_outputs']:
                assert sample['received_monotonic'] <= sample['decoded_monotonic'] <= sample['processing_monotonic'] <= sample['encoded_monotonic']
        assert events[-1]["video_decoder"] == "h264_nvdec"
        assert events[-1]["video_encoder"] == "h264_nvenc"
    software_mask.assert_not_called()
    if not debug:
        assert all('stream' not in event.get('stats', {}) and 'timings' not in event for event in events)
    assert stop.is_set()
    if fault != "decode":
        assert closed.is_set()


def test_live_pacing_can_be_disabled_without_changing_masked_output(monkeypatch):
    test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(
        monkeypatch, None, False, False, True, pacing=False)


def test_slow_encoder_does_not_squeeze_following_final_packets_together(monkeypatch):
    test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(
        monkeypatch, None, False, False, True, slow_first=True)


def test_muted_rtmp_with_audio_repairs_video_clock_on_cuda_path(monkeypatch):
    """带音轨的 RTMP 静音后仍可修复重复视频 PTS，保留所有独立画面。"""
    test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(
        monkeypatch, fault=None, delayed=False, repair=True, debug=True, muted=True,
    )


def test_audible_rtmp_repairs_video_clock_before_cuda_masking(monkeypatch):
    """有声帧在 GPU 遮罩及编码前完成时钟修复，原始 PTS 仍用于诊断。"""
    test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(
        monkeypatch, fault=None, delayed=False, repair=True, debug=True, with_audio=True,
    )


def test_stream_without_experimental_headers_uses_route_a_on_cuda_path(monkeypatch):
    test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(
        monkeypatch, fault=None, delayed=False, repair=False, debug=True, default_schedule=True,
    )
