import copy
import json
from pathlib import Path
import runpy

import pytest


def evaluate(rows):
    module = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/evaluate-stream-metrics.py'))
    return module['summarize'](rows)


def observation(count=200, end=10, attempt='one', start=100):
    outputs = [dict(sequence=i + 1, source_pts=str(i / 20), received_monotonic=start + i / 20,
                    encoded_monotonic=start + i / 20 + .01, latency_seconds=.01) for i in range(count)]
    return dict(task=dict(id='task', state='running', stats=dict(processed_frames=count, dropped_frames=0,
        stream=dict(attempt_id=attempt, started_monotonic=start, sampled_monotonic=start + end,
                    input_frames=count, encoded_frames=count, encoded_packets=count, unmatched_packets=0,
                    missing_receive_frames=0, pending_encoder_frames=0, queue_frames=0,
                    audio_queue_bytes=0, failed_frames=0, protected_frames=0, recent_outputs=outputs))))


def test_polling_overlap_does_not_inflate_output_and_empty_windows_remain():
    row = observation(end=20)
    result = evaluate([row, copy.deepcopy(row)])['attempts'][0]
    assert result['sample_count'] == 200 and result['evidence_errors'] == []
    assert result['fps_per_second'] == [20] * 10 + [0] * 10
    assert [w['fps'] for w in result['ten_second_windows']] == [20, 0]
    assert not result['all_complete_10s_windows_at_least_20_fps']
    assert result['latency_seconds']['p99'] == .01


def test_missing_samples_and_reconnects_are_not_a_continuous_run():
    row = observation()
    del row['task']['stats']['stream']['recent_outputs'][1]
    result = evaluate([row, observation(attempt='two', start=130)])
    assert len(result['attempts']) == 2
    assert 'missing_output_sequences' in result['attempts'][0]['evidence_errors']
    assert result['attempts'][0]['latency_thresholds_met'] is None
    assert all(a['observed_duration_seconds'] == 10 for a in result['attempts'])


def test_unknown_receive_time_cannot_pass_latency_gate():
    row = observation()
    stream = row['task']['stats']['stream']
    stream['recent_outputs'][0].update(received_monotonic=None, latency_seconds=None)
    stream['missing_receive_frames'] = 1
    result = evaluate([row])['attempts'][0]
    assert 'missing_receive_time' in result['evidence_errors']
    assert result['latency_thresholds_met'] is None


def test_no_rounding_up_a_below_20_fps_window():
    result = evaluate([observation(count=199)])['attempts'][0]
    assert result['ten_second_windows'][0]['fps'] == 19.9
    assert result['all_complete_10s_windows_at_least_20_fps'] is False


@pytest.mark.parametrize('change', ['conflict', 'source_clock', 'negative_latency'])
def test_invalid_sample_evidence_is_rejected(change):
    row = observation()
    modified = copy.deepcopy(row)
    samples = modified['task']['stats']['stream']['recent_outputs']
    if change == 'conflict':
        samples[0]['encoded_monotonic'] += .001
    elif change == 'source_clock':
        samples[1]['source_pts'] = samples[0]['source_pts']
    else:
        samples[0]['latency_seconds'] = -1
    rows = [row, modified] if change == 'conflict' else [modified]
    assert evaluate(rows)['attempts'][0]['evidence_errors']


def test_latency_above_threshold_is_not_rounded_down():
    row = observation(count=100)
    for sample in row['task']['stats']['stream']['recent_outputs']:
        sample['latency_seconds'] = 2.0001
        sample['encoded_monotonic'] = sample['received_monotonic'] + 2.0001
    result = evaluate([row])['attempts'][0]
    assert result['evidence_errors'] == []
    assert result['latency_seconds']['p95'] == 2.0001
    assert result['latency_thresholds_met'] is False


def test_cli_saves_report_and_missing_metrics_are_explicit(tmp_path, monkeypatch):
    row = observation()
    del row['task']['stats']['stream']
    source, target = tmp_path / 'observations.jsonl', tmp_path / 'report.json'
    source.write_text(json.dumps(row) + '\n', encoding='utf-8')
    script = Path(__file__).resolve().parents[2] / 'scripts/evaluate-stream-metrics.py'
    monkeypatch.setattr('sys.argv', [str(script), str(source), str(target)])
    runpy.run_path(str(script), run_name='__main__')
    report = json.loads(target.read_text('utf-8'))
    assert not report['attempts'] and report['evidence_errors'] == ['no_stream_metrics']
