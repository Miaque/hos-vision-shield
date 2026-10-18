import threading
from types import SimpleNamespace
from fractions import Fraction

import pytest

from vision_shield.media.streams import StreamBuffer


def test_overload_discards_old_frames_but_keeps_only_bounded_audio():
    buffer = StreamBuffer(2, 2, 4, threading.Event())
    for index in range(4):
        buffer.frame(SimpleNamespace(pts=index, time_base=Fraction(1)))
    assert [frame.pts for frame in buffer.frames] == [2, 3] and buffer.dropped == 2
    packet = SimpleNamespace(size=3, pts=0, time_base=Fraction(1))
    buffer.audio_packet(packet)
    with pytest.raises(BufferError):
        buffer.audio_packet(packet)


def test_take_uses_newest_frame_and_preserves_audio_clock():
    buffer = StreamBuffer(2, 2, 100, threading.Event())
    for pts in [1, 2]:
        buffer.frame(SimpleNamespace(pts=pts, time_base=Fraction(1)))
    packets = [SimpleNamespace(size=3, pts=pts, time_base=Fraction(1)) for pts in [1, 2, 3]]
    for packet in packets:
        buffer.audio_packet(packet)
    frame, audio = buffer.take()
    assert frame.pts == 2 and buffer.dropped == 1 and not buffer.frames
    assert audio == packets[:2] and list(buffer.audio) == packets[2:] and buffer.audio_size == 3


@pytest.mark.parametrize("pts,time_base", [(None, Fraction(1, 90000)), (5940, None)])
def test_unclocked_live_frame_is_dropped_without_restarting_valid_stream(pts, time_base):
    buffer = StreamBuffer(2, 16, 100, threading.Event())
    buffer.frame(SimpleNamespace(pts=pts, time_base=time_base))
    assert not buffer.frames and buffer.dropped == 1
    valid = SimpleNamespace(pts=5940, time_base=Fraction(1, 90000))
    buffer.frame(valid)
    assert buffer.take() == (valid, [])
    assert buffer.dropped == 1
