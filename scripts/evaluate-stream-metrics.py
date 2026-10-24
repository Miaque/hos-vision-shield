"""Evaluate saved per-task API observations; this does not certify output quality."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path


def percentiles(values):
    ordered = sorted(values)
    return {f'p{p}': ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)] if ordered else None
            for p in (50, 95, 99)}


def summarize(observations):
    attempts, states = {}, []
    for row in observations:
        task = row['task']
        state = (task['id'], task['state'])
        if not states or (states[-1]['task_id'], states[-1]['state']) != state:
            states.append(dict(task_id=state[0], state=state[1], observed_unix=row.get('unix')))
        stream = task['stats'].get('stream')
        if not stream:
            continue
        key = (task['id'], stream['attempt_id'])
        attempt = attempts.setdefault(key, dict(latest=stream, stats=task['stats'], samples={}, errors=set()))
        errors = attempt['errors']
        if stream['started_monotonic'] != attempt['latest']['started_monotonic']:
            errors.add('attempt_start_changed')
        if stream['sampled_monotonic'] >= attempt['latest']['sampled_monotonic']:
            if stream['encoded_frames'] < attempt['latest']['encoded_frames']:
                errors.add('output_counter_regressed')
            attempt.update(latest=stream, stats=task['stats'])
        for sample in stream['recent_outputs']:
            old = attempt['samples'].setdefault(sample['sequence'], sample)
            if old != sample:
                errors.add('conflicting_output_sample')

    reports = []
    for (task_id, attempt_id), attempt in attempts.items():
        latest, errors = attempt['latest'], attempt['errors']
        samples = [sample for _, sample in sorted(attempt['samples'].items())]
        if list(sorted(attempt['samples'])) != list(range(1, latest['encoded_frames'] + 1)):
            errors.add('missing_output_sequences')
        start, end = latest['started_monotonic'], latest['sampled_monotonic']
        if not math.isfinite(end - start) or end < start:
            raise ValueError('Invalid observation clock')
        seconds = [[] for _ in range(math.floor(end - start))]
        latencies, drifts, timeline_drifts = [], [], []
        previous_source, previous_timeline, previous_output, previous_offset = None, None, None, 0
        repairs = raw_anomalies = 0
        for sample in samples:
            source, output = Fraction(sample['source_pts']), sample['encoded_monotonic']
            timeline = Fraction(sample.get('timeline_pts') or sample['source_pts'])
            offset = timeline - source
            repaired = sample.get('timestamp_repaired', False)
            if not math.isfinite(output) or output < start or output > end:
                errors.add('invalid_output_clock')
                continue
            if previous_timeline is not None and (timeline <= previous_timeline or output < previous_output):
                errors.add('nonincreasing_frame_clock')
            raw_anomalies += previous_source is not None and source <= previous_source
            if offset != previous_offset and not repaired:
                errors.add('unreported_clock_offset_change')
            if repaired and (previous_timeline is None or source + previous_offset > previous_timeline or offset <= previous_offset):
                errors.add('invalid_timestamp_repair')
            repairs += repaired
            previous_source, previous_output = source, output
            previous_timeline, previous_offset = timeline, offset
            received, latency = sample['received_monotonic'], sample['latency_seconds']
            if received is None or latency is None:
                errors.add('missing_receive_time')
            elif (not math.isfinite(received) or not math.isfinite(latency) or latency < 0
                  or not math.isclose(output - received, latency, abs_tol=1e-6)):
                errors.add('invalid_latency')
            else:
                latencies.append(latency)
            index = math.floor(output - start)
            if index < len(seconds):
                seconds[index].append(sample)
            drifts.append(output - samples[0]['encoded_monotonic']
                          - float(source - Fraction(samples[0]['source_pts'])))
            timeline_drifts.append(output - samples[0]['encoded_monotonic']
                                   - float(timeline - Fraction(samples[0].get('timeline_pts') or samples[0]['source_pts'])))
        if repairs != latest.get('timestamp_repairs', 0):
            errors.add('timestamp_repair_counter_mismatch')
        for key in ('unmatched_packets', 'missing_receive_frames', 'failed_frames', 'protected_frames'):
            if latest[key]:
                errors.add(key)
        if attempt['stats']['dropped_frames']:
            errors.add('dropped_frames')
        windows = []
        for index in range(len(seconds) // 10):
            frames = [sample for bucket in seconds[index * 10:(index + 1) * 10] for sample in bucket]
            windows.append(dict(start_seconds=index * 10, frames=len(frames), fps=len(frames) / 10,
                latency_seconds=percentiles([s['latency_seconds'] for s in frames
                    if s['latency_seconds'] is not None and math.isfinite(s['latency_seconds']) and s['latency_seconds'] >= 0])))
        distribution = percentiles(latencies)
        reports.append(dict(task_id=task_id, attempt_id=attempt_id, observed_duration_seconds=end - start,
            window_origin_monotonic=start, sample_count=len(samples), evidence_errors=sorted(errors),
            counters={key: value for key, value in latest.items() if key != 'recent_outputs'},
            processed_frames=attempt['stats']['processed_frames'], dropped_frames=attempt['stats']['dropped_frames'],
            fps_per_second=[len(bucket) for bucket in seconds], ten_second_windows=windows,
            latency_seconds=distribution, source_to_output_drift_seconds=percentiles(drifts),
            source_to_output_drift_last_seconds=drifts[-1] if drifts else None,
            raw_nonincreasing_timestamps=raw_anomalies,
            timeline_to_output_drift_seconds=percentiles(timeline_drifts),
            timeline_to_output_drift_last_seconds=timeline_drifts[-1] if timeline_drifts else None,
            all_complete_10s_windows_at_least_20_fps=(all(w['fps'] >= 20 for w in windows)
                                                    if windows and not errors else None),
            latency_thresholds_met=(distribution['p95'] <= 2 and distribution['p99'] <= 5
                                    if latencies and not errors else None)))
    return dict(scope='Per-attempt timing evidence only; quality, hardware, dual overlap and recovery require separate review',
                quantiles='nearest rank', window_rule='Non-overlapping complete windows from attempt start; startup and zero-output windows included',
                task_states=states, attempts=reports, evidence_errors=[] if reports else ['no_stream_metrics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.observations.open(encoding='utf-8') as source:
        report = summarize(json.loads(line) for line in source if line.strip())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
