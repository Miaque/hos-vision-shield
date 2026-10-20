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


def test_live_encoding_keeps_its_low_latency_options():
    video = SimpleNamespace(codec_context=SimpleNamespace())
    writer = SimpleNamespace(add_stream=Mock(return_value=video))
    frame = SimpleNamespace(width=320, height=240, time_base=Fraction(1, 90000))
    add_video(writer, frame, SimpleNamespace(average_rate=Fraction(25)), live=True)
    assert video.options == {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
    assert video.codec_context.gop_size == 1
