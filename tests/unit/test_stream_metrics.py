from fractions import Fraction
from types import SimpleNamespace

import pytest

from vision_shield.media.stream_metrics import StreamMetrics


def frame(pts):
    return SimpleNamespace(pts=pts, time_base=Fraction(1, 1000))


def test_only_unique_written_frames_count_and_keep_receive_latency():
    metrics = StreamMetrics()
    metrics.submit(frame(40), 10.0, Fraction(90040, 1000))
    assert metrics.snapshot()['encoded_frames'] == 0
    metrics.written(Fraction(40, 1000), 10.25)
    metrics.written(Fraction(40, 1000), 10.30)  # Duplicate packet is not another input frame.
    metrics.written(Fraction(80, 1000), 10.35)  # Unknown packet must not inflate throughput.
    result = metrics.snapshot()
    assert result['encoded_frames'] == 1 and result['encoded_packets'] == 3
    assert result['unmatched_packets'] == 2 and result['pending_encoder_frames'] == 0
    assert result['timestamp_repairs'] == 0
    assert result['recent_outputs'] == [dict(sequence=1, source_pts='2251/25', timeline_pts='2251/25', timestamp_repaired=False,
        received_monotonic=10.0, encoded_monotonic=10.25, latency_seconds=0.25)]


def test_missing_receive_clock_is_explicit_and_samples_are_bounded():
    metrics = StreamMetrics()
    for index in range(600):
        metrics.submit(frame(index), None, Fraction(index, 1000))
        metrics.written(Fraction(index, 1000), float(index))
    result = metrics.snapshot()
    assert result['encoded_frames'] == result['missing_receive_frames'] == 600
    assert len(result['recent_outputs']) == 512
    assert result['recent_outputs'][0]['sequence'] == 89
    assert result['recent_outputs'][-1]['latency_seconds'] is None
    assert StreamMetrics().snapshot()['attempt_id'] != result['attempt_id']


def test_encoder_backlog_cannot_grow_without_bound():
    metrics = StreamMetrics()
    for index in range(512):
        metrics.submit(frame(index), 1.0, Fraction(index))
    with pytest.raises(BufferError, match='Encoder backlog'):
        metrics.submit(frame(512), 1.0, Fraction(512))
    assert metrics.snapshot()['encoded_frames'] == 0
