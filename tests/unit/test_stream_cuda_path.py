"""CPU contract checks; actual NVDEC/masking/NVENC still require the GPU server."""

from fractions import Fraction
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import av
import torch

from vision_shield.config import Settings
from vision_shield.media import cuda_frames, streams


@pytest.mark.parametrize("fault", [None, "decode", "mask", "encode", "mux"])
@pytest.mark.parametrize("delayed", [False, True])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize('debug', [False, True])
def test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed(monkeypatch, fault, delayed, repair, debug):
    source_video = SimpleNamespace(width=64, height=48, codec_context=SimpleNamespace(name="h264"))
    frames = [SimpleNamespace(width=64, height=48, pts=900000 + 3600 * i, time_base=Fraction(1, 90000))
              for i in range(3)]
    if repair:
        frames[-1].pts = frames[-2].pts
    threads, masked, encoded = {}, [], []
    closed = threading.Event()
    accelerator = object()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: threads.update(selected=device))

    class Reader:
        streams = SimpleNamespace(video=[source_video], audio=[])

        def demux(self):
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

        def masked(self, frame, redactor):
            assert redactor.detection_interval == 3 and redactor.tracking == "flow"
            if fault == "mask":
                raise RuntimeError("mask failed")
            masked.append(frame.pts)
            return SimpleNamespace(**vars(frame))

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def mux(self, packet):
            if fault == "mux":
                raise RuntimeError("mux failed")
            encoded.append(packet.pts)
            packet.pts = -1  # Muxers may rescale/mutate the packet after accepting it.

    def open_endpoint(url, *args, **kwargs):
        if url == "rtsp://camera/input":
            threads["decode"] = threading.get_ident()
            assert kwargs["hwaccel"] is accelerator
            assert kwargs["options"]["allowed_media_types"] == "video"
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
    config = {**Settings(_env_file=None, debug=debug).model_dump(), "inference_device": "cuda",
              "detection_interval": 3, "tracking": "flow"}
    events = []
    stop = threading.Event()

    def process():
        streams.process_stream("rtsp://camera/input", "rtsp://output/test", object(), config, stop, events.append)

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
        assert threads["context"] == threads["decode"] != threading.get_ident()
        assert add_video.call_args.kwargs == {"live": True, "device": "cuda"}
        assert events[-1]["stats"]["processed_frames"] == 3
        if debug:
            assert events[-1]["stats"]["stream"]["encoded_frames"] == 3
            assert events[-1]["stats"]["stream"]["missing_receive_frames"] == 0
            assert events[-1]["stats"]["stream"]["timestamp_repairs"] == int(repair)
            assert [s['source_pts'] for s in events[-1]['stats']['stream']['recent_outputs']] == [
                '10', '251/25', '251/25' if repair else '252/25']
        assert events[-1]["video_decoder"] == "h264_nvdec"
        assert events[-1]["video_encoder"] == "h264_nvenc"
    software_mask.assert_not_called()
    if not debug:
        assert all('stream' not in event.get('stats', {}) and 'timings' not in event for event in events)
    assert stop.is_set()
    if fault != "decode":
        assert closed.is_set()
