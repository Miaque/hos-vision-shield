import threading
from types import SimpleNamespace
from fractions import Fraction

import pytest

from vision_shield.media.streams import StreamBuffer


def test_overload_waits_for_capacity_without_losing_frames():
    buffer = StreamBuffer(2, 2, 4, threading.Event())
    for index in range(2):
        buffer.frame(SimpleNamespace(pts=index, time_base=Fraction(1)))
    entered, finished = threading.Event(), threading.Event()

    def produce():
        entered.set()
        buffer.frame(SimpleNamespace(pts=2, time_base=Fraction(1)))
        finished.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert entered.wait(1)
        assert not finished.wait(0.1)
        assert [frame.pts for frame in buffer.frames] == [0, 1]
        assert buffer.take()[0].pts == 0
        assert finished.wait(1)
        assert [buffer.take()[0].pts for _ in range(2)] == [1, 2]
        assert buffer.dropped == 0
    finally:
        buffer.stop.set()
        producer.join(timeout=1)
    assert not producer.is_alive()


def test_stop_releases_producer_waiting_for_capacity():
    buffer = StreamBuffer(1, 2, 4, threading.Event())
    buffer.frame(SimpleNamespace(pts=0, time_base=Fraction(1)))
    producer = threading.Thread(target=buffer.frame, args=(SimpleNamespace(pts=1, time_base=Fraction(1)),), daemon=True)
    producer.start()
    try:
        producer.join(timeout=0.1)
        assert producer.is_alive()
    finally:
        buffer.stop.set()
        producer.join(timeout=1)
    assert not producer.is_alive()
    assert [frame.pts for frame in buffer.frames] == [0]


def test_audio_remains_bounded():
    buffer = StreamBuffer(2, 2, 4, threading.Event())
    packet = SimpleNamespace(size=3, pts=0, time_base=Fraction(1))
    buffer.audio_packet(packet)
    with pytest.raises(BufferError):
        buffer.audio_packet(packet)


def test_receive_timestamp_survives_queueing_and_input_count_includes_invalid_clock():
    buffer = StreamBuffer(2, 2, 4, threading.Event())
    buffer.frame(SimpleNamespace(pts=1, time_base=Fraction(1)), received_at=10.0)
    buffer.frame(SimpleNamespace(pts=None, time_base=Fraction(1)), received_at=11.0)
    buffer.take()
    assert buffer.last_received == 10.0
    assert buffer.input_frames == 2 and buffer.dropped == 1


def test_decode_release_clock_stays_with_each_fifo_frame():
    buffer = StreamBuffer(2, 2, 4, threading.Event())
    for pts, received, decoded in [(1, 10.0, 16.0), (2, 10.04, 16.25)]:
        buffer.frame(SimpleNamespace(pts=pts, time_base=Fraction(1)), received_at=received, decoded_at=decoded)
    for pts, received, decoded in [(1, 10.0, 16.0), (2, 10.04, 16.25)]:
        assert buffer.take()[0].pts == pts
        assert (buffer.last_received, buffer.last_decoded) == (received, decoded)


def test_take_uses_fifo_and_preserves_audio_clock():
    buffer = StreamBuffer(2, 2, 100, threading.Event())
    for pts in [1, 2]:
        buffer.frame(SimpleNamespace(pts=pts, time_base=Fraction(1)))
    packets = [SimpleNamespace(size=3, pts=pts, time_base=Fraction(1)) for pts in [1, 2, 3]]
    for packet in packets:
        buffer.audio_packet(packet)
    frame, audio = buffer.take()
    assert frame.pts == 1 and buffer.dropped == 0
    assert audio == packets[:1] and list(buffer.audio) == packets[1:] and buffer.audio_size == 6
    frame, audio = buffer.take()
    assert frame.pts == 2 and buffer.dropped == 0 and not buffer.frames
    assert audio == packets[1:2] and list(buffer.audio) == packets[2:] and buffer.audio_size == 3


@pytest.mark.parametrize("pts,time_base", [(None, Fraction(1, 90000)), (5940, None)])
def test_unclocked_live_frame_is_dropped_without_restarting_valid_stream(pts, time_base):
    buffer = StreamBuffer(2, 16, 100, threading.Event())
    buffer.frame(SimpleNamespace(pts=pts, time_base=time_base))
    assert not buffer.frames and buffer.dropped == 1
    valid = SimpleNamespace(pts=5940, time_base=Fraction(1, 90000))
    buffer.frame(valid)
    assert buffer.take() == (valid, [])
    assert buffer.dropped == 1
