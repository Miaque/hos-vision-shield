from fractions import Fraction
from types import SimpleNamespace

import pytest

from vision_shield.media.stream_metrics import StreamMetrics, vcl_digest


def frame(pts):
    return SimpleNamespace(pts=pts, time_base=Fraction(1, 1000))


def test_attempt_origin_keeps_model_loading_as_zero_output_windows(monkeypatch):
    import runpy
    from pathlib import Path
    from vision_shield.media import stream_metrics

    monkeypatch.setattr(stream_metrics.time, 'monotonic', lambda: 130.0)
    metrics = StreamMetrics(started=100.0, debug=True)
    metrics.submit(frame(40), 115.0, Fraction(40, 1000))
    metrics.written(Fraction(40, 1000), 115.25)
    stream = metrics.snapshot()
    assert stream['started_monotonic'] == 100.0
    stream.update(input_frames=1, queue_frames=0, audio_queue_bytes=0, failed_frames=0, protected_frames=0)
    row = dict(task=dict(id='task', state='running', stats=dict(
        processed_frames=1, dropped_frames=0, stream=stream)))
    evaluate = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/evaluate-stream-metrics.py'))['summarize']
    report = evaluate([row])['attempts'][0]
    assert report['evidence_errors'] == []
    assert report['observed_duration_seconds'] == 30.0
    assert [window['fps'] for window in report['ten_second_windows']] == [0.0, 0.1, 0.0]


def test_only_unique_written_frames_count_and_keep_receive_latency():
    metrics = StreamMetrics(debug=True)
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
        encoded_bytes=None, encoded_vcl_sha256=None,
        received_monotonic=10.0, encoded_monotonic=10.25, latency_seconds=0.25)]


def test_vcl_hash_ignores_annexb_prefixes_parameter_sets_and_padding_but_detects_image_byte_changes():
    first = b'\x00\x00\x00\x01\x67sps\x00\x00\x00\x01\x65image\x00\x00'
    same = b'\x00\x00\x01\x68pps\x00\x00\x01\x65image'
    assert vcl_digest(first) == vcl_digest(same)
    assert vcl_digest(first) != vcl_digest(same + b'x')
    assert vcl_digest(b'\x00\x00\x01\x67sps') is None
    assert vcl_digest(b'') is None


def test_encoded_packet_evidence_survives_output_sample():
    metrics = StreamMetrics(debug=True)
    metrics.submit(frame(40), 10.0, Fraction(1))
    digest = vcl_digest(b'\x00\x00\x01\x65image')
    metrics.written(Fraction(40,1000),10.25,encoded_bytes=9,encoded_vcl_sha256=digest)
    sample = metrics.snapshot()['recent_outputs'][0]
    assert sample['encoded_bytes'] == 9 and sample['encoded_vcl_sha256'] == digest


def test_missing_receive_clock_is_explicit_and_samples_are_bounded():
    metrics = StreamMetrics(debug=True)
    for index in range(600):
        metrics.submit(frame(index), None, Fraction(index, 1000))
        metrics.written(Fraction(index, 1000), float(index))
    result = metrics.snapshot()
    assert result['encoded_frames'] == result['missing_receive_frames'] == 600
    assert len(result['recent_outputs']) == 512
    assert result['recent_outputs'][0]['sequence'] == 89
    assert result['recent_outputs'][-1]['latency_seconds'] is None
    assert StreamMetrics().snapshot()['attempt_id'] != result['attempt_id']


@pytest.mark.parametrize('debug', [False, True])
def test_encoder_backlog_cannot_grow_without_bound(debug):
    metrics = StreamMetrics(debug=debug)
    for index in range(512):
        metrics.submit(frame(index), 1.0, Fraction(index))
    with pytest.raises(BufferError, match='Encoder backlog'):
        metrics.submit(frame(512), 1.0, Fraction(512))
    assert metrics.snapshot()['encoded_frames'] == 0


def test_production_keeps_encoder_guards_without_recording_samples():
    metrics = StreamMetrics(debug=False)
    metrics.submit(frame(40), None, object())
    with pytest.raises(BufferError, match='Duplicate encoder input clock'):
        metrics.submit(frame(40), None, object())
    metrics.written(Fraction(40, 1000), 10.25)
    assert not metrics.outputs and not metrics.pending
    assert metrics.frames == 1
