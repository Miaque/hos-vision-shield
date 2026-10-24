from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vision_shield.media.video import add_video


@pytest.mark.parametrize("source_rate", [None, 0, -1, 2_714_928])
def test_file_encoding_keeps_quality_target_and_uses_available_source_rate(source_rate):
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=1920, height=536, time_base=Fraction(1, 20000))
    source = SimpleNamespace(average_rate=Fraction(20), bit_rate=source_rate)
    add_video(writer, frame, source)
    assert video.options["crf"] == "23" and video.options["preset"] == "slow"
    assert video.time_base == video.codec_context.time_base == frame.time_base
    writer.add_stream.assert_called_once_with("libx264", rate=Fraction(20))
    if source_rate and source_rate > 0:
        assert video.options["maxrate"] == str(round(source_rate * 1.15))
        assert video.options["bufsize"] == str(round(source_rate * 1.15) * 2)
    else:
        assert "maxrate" not in video.options and "bufsize" not in video.options


@pytest.mark.parametrize('rate,gop', [(Fraction(25), 25), (None, 25), (Fraction(30000, 1001), 30)])
def test_live_encoding_keeps_its_low_latency_options(rate, gop):
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=320, height=240, time_base=Fraction(1, 90000))
    add_video(writer, frame, SimpleNamespace(average_rate=rate), live=True)
    assert video.options == {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
    assert video.codec_context.gop_size == gop


def test_cuda_file_encoding_uses_nvenc_on_the_selected_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=3840, height=1080, time_base=Fraction(1, 16000))
    source = SimpleNamespace(average_rate=Fraction(15), bit_rate=2_000_000)
    add_video(writer, frame, source, device="cuda")
    writer.add_stream.assert_called_once_with("h264_nvenc", rate=Fraction(15))
    assert video.options["gpu"] == "2" and video.options["rc"] == "vbr"
    assert video.options["preset"] == "p6" and video.options["tune"] == "hq"
    assert video.options["cq"] == "19" and "crf" not in video.options
    assert video.options["maxrate"] == "2300000" and video.options["bufsize"] == "4600000"
    assert video.time_base == video.codec_context.time_base == frame.time_base


def test_live_nvenc_omits_rate_cap_when_not_requested(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=3840, height=1080, time_base=Fraction(1, 90000))
    add_video(writer, frame, SimpleNamespace(average_rate=Fraction(15)), live=True, device="cuda")
    assert "maxrate" not in video.options and "bufsize" not in video.options


@pytest.mark.parametrize("live", [False, True])
def test_unavailable_nvenc_is_never_replaced_with_cpu_encoding(live):
    writer = SimpleNamespace(add_stream=Mock(side_effect=RuntimeError("NVENC unavailable")))
    frame = SimpleNamespace(width=320, height=240, time_base=Fraction(1, 15))
    with pytest.raises(RuntimeError, match="NVENC unavailable"):
        add_video(writer, frame, SimpleNamespace(average_rate=Fraction(15)), device="cuda", live=live)
    writer.add_stream.assert_called_once_with("h264_nvenc", rate=Fraction(15))


@pytest.mark.parametrize("live", [False, True])
def test_cuda_frames_go_directly_to_nvenc_with_the_selected_context_and_color_metadata(monkeypatch, live):
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=3840, height=1080, time_base=Fraction(1, 16000),
                            format=SimpleNamespace(name="cuda"), colorspace=1, color_range=2,
                            color_trc=1, color_primaries=1)
    source = SimpleNamespace(average_rate=Fraction(15), bit_rate=2_000_000)
    add_video(writer, frame, source, device="cuda", live=live)
    call = writer.add_stream.call_args
    assert call.args == ("h264_nvenc",) and call.kwargs["rate"] == Fraction(15)
    hwaccel = call.kwargs["hwaccel"]
    assert hwaccel.device_id == 2 and hwaccel.is_hw_owned and not hwaccel.allow_software_fallback
    assert hwaccel.options == {"primary_ctx": "0", "current_ctx": "1"}
    assert video.pix_fmt == "cuda" and video.options["gpu"] == "2"
    assert video.codec_context.colorspace == 1 and video.codec_context.color_range == 2
    assert video.options["tune"] == ("ull" if live else "hq")
    assert video.options["bf"] == ("0" if live else "3")
    if live:
        assert video.codec_context.gop_size == 15


def test_live_nvenc_caps_rate_while_retaining_cq_resolution_and_clock(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=3840, height=1080, time_base=Fraction(1, 90000))
    add_video(writer, frame, SimpleNamespace(average_rate=None), live=True, device='cuda', max_bitrate=20_000_000)
    assert video.width == 3840 and video.height == 1080 and video.bit_rate == 0
    assert video.time_base == video.codec_context.time_base == frame.time_base
    assert video.codec_context.gop_size == 25
    assert video.options == dict(preset='p6', tune='ull', rc='vbr', cq='19', gpu='2', bf='0',
                                 maxrate='20000000', bufsize='20000000')
