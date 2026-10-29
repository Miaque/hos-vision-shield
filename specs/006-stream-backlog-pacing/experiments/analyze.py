"""Analyze HTTP evidence without mixing client and server monotonic clocks."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASELINE = '1c9be543c30fed7a2f4263b85c447ba079d438f8'


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0)
    def q(f):
        i = (len(values) - 1) * f
        low = int(i)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (i - low)
    return dict(n=len(values), mean=sum(values)/len(values), p50=q(.5), p95=q(.95),
                p99=q(.99), min=values[0], max=values[-1])


def union(intervals):
    end, total = float('-inf'), 0.
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def timings(row):
    return {name.removeprefix('task-'): float(value)/1000
            for item in row.get('headers', {}).get('server-timing', '').split(', ')
            if ';dur=' in item for name, value in [item.split(';dur=')]}


def intervals(rows, clock, group):
    gaps = [b[clock]-a[clock] for a, b in zip(rows, rows[1:])
            if group(a) == group(b) and b['sequence'] == a['sequence']+1]
    return dict(seconds=distribution(gaps), over_500ms=sum(x > .5 for x in gaps),
                over_1000ms=sum(x > 1 for x in gaps),
                sequence_gaps=sum(b['sequence'] != a['sequence']+1 for a, b in zip(rows, rows[1:])
                                  if group(a) == group(b)))


def low_fps(times, seconds):
    # Fixed wall-clock windows anchored on the first received frame; omit partial tail.
    if len(times) < 2:
        return dict(n=0)
    n = int((times[-1]-times[0])/seconds)
    counts = [0]*n
    for t in times:
        index = int((t-times[0])/seconds)
        if index < n:
            counts[index] += 1
    return distribution([count/seconds for count in counts])


def analyze(folder):
    rows = [json.loads(line) for line in (folder/'live.jsonl').read_text(encoding='utf-8').splitlines()]
    runtime = {r['attempt_id']: r['data'] for r in rows if r['kind'] == 'runtime'}
    checks = {}
    for aid, meta in runtime.items():
        checks[aid] = {name: hashlib.sha256(subprocess.check_output(
            ['git', 'show', f'{BASELINE}:src/vision_shield/{name.replace(".", "/")}.py'])).hexdigest() == digest
            for name, digest in meta['source_sha256'].items()}
        assert len(checks[aid]) == 17 and all(checks[aid].values()), 'Unverified runtime'
        assert meta['output_mux_options'].get('max_interleave_delta') == '100000'
    tasks = [r for r in rows if r['kind'] == 'task']
    grouped = defaultdict(list)
    for row in tasks:
        grouped[row['data']['stats']['stream']['attempt_id']].append(row)
    assert runtime and grouped and all(aid in runtime for aid in grouped), 'Missing runtime identity'
    frames = [r for r in rows if r['kind'] == 'frame']
    operations = [r for r in rows if r['kind'] == 'operation']
    summaries = {}
    for aid, group in grouped.items():
        a, b = [r['data']['stats']['stream'] for r in (group[0], group[-1])]
        t0, t1 = a['sampled_monotonic'], b['sampled_monotonic']
        before, after = timings(group[0]), timings(group[-1])
        delta = {key: val-before.get(key, 0) for key, val in after.items()}
        output_count = b['encoded_frames'] - a['encoded_frames']
        selected = sorted((r for r in frames if r['attempt_id'] == aid and t0 <= r['encoded_monotonic'] <= t1),
                          key=lambda r: r['sequence'])
        ops = [r for r in operations if r['attempt_id'] == aid
               and r['started_monotonic'] >= t0 and r['ended_monotonic'] <= t1]
        progress = None
        if len(selected) > 1:
            first, last = selected[0], selected[-1]
            wall = last['encoded_monotonic']-first['encoded_monotonic']
            media = float(Fraction(last['source_pts'])-Fraction(first['source_pts']))
            jumps = [dict(sequence=y['sequence'], delta_seconds=float(Fraction(y['source_pts'])-Fraction(x['source_pts'])))
                     for x, y in zip(selected, selected[1:])
                     if y['sequence'] == x['sequence']+1 and not 0 < Fraction(y['source_pts'])-Fraction(x['source_pts']) <= Fraction(1, 5)]
            progress = dict(wall_seconds=wall, source_media_seconds=media,
                            lag_growth_seconds=wall-media, lag_growth_per_second=1-media/wall,
                            pts_jumps=jumps)
        full = [r['data']['stats']['stream']['queue_frames'] == 2 for r in group]
        proc = [r for r in ops if r['operation_kind'] == 'process']
        compute_union = union([(r['started_monotonic'], r['ended_monotonic']) for r in ops
                               if r['operation_kind'] in ('process', 'render') or r['operation_kind'].startswith('lookahead')])
        exclusive = {key: union([(r['started_monotonic'], r['ended_monotonic']) for r in ops
                                if r['operation_kind'] == key])
                     for key in sorted({r['operation_kind'] for r in ops})}
        nonpacing = delta.get('frame', 0) + delta.get('encode', 0)
        summaries[aid] = dict(window_seconds=t1-t0, output_frames=output_count,
            output_fps=output_count/(t1-t0) if t1 > t0 else None, progress=progress,
            queue_full_sample_fraction=sum(full)/len(full),
            queue_frames=distribution([r['data']['stats']['stream']['queue_frames'] for r in group]),
            lookahead_frames=distribution([r['data']['stats']['stream']['lookahead_frames'] for r in group]),
            timing_delta_seconds=delta,
            nonpacing_ms_per_output=nonpacing/output_count*1000 if output_count else None,
            optimistic_nonpacing_fps=output_count/nonpacing if nonpacing else None,
            source_backpressure_fraction=delta.get('source-backpressure', 0)/(t1-t0) if t1 > t0 else None,
            process_seconds=distribution([r['seconds'] for r in proc]),
            observed_compute_union_seconds=compute_union, observed_operation_unions=exclusive,
            mux_intervals=intervals(selected, 'encoded_monotonic', lambda r: aid),
            repairs=sum(r['timestamp_repaired'] for r in selected),
            counters={key: b[key]-a[key] for key in ('unmatched_packets', 'failed_frames', 'missing_receive_frames')},
            process_frame_sequence_gaps=sum(y['frame_sequence'] != x['frame_sequence']+1 for x, y in zip(proc, proc[1:])))
    internal = [r for r in rows if r['kind'] == 'internal_frame']
    identities = [r['data']['clock'] for r in rows if r['kind'] == 'internal_state']
    assert all(identity == meta['clock_identity'] for identity in identities for meta in runtime.values())
    rtp = intervals(internal, 'received_monotonic', lambda r: (r['observer_id'], r['connection']))
    times = [r['received_monotonic'] for r in internal]
    rtp.update(low_1s_fps=low_fps(times, 1), low_5s_fps=low_fps(times, 5),
               observed_frames=len(times), effective_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None)
    attempts = {r['attempt_id']: r for row in rows if row['kind'] == 'attempts'
                for r in row['data'].get('attempts', []) if 'attempt_id' in r}
    ordered = list(attempts.values())
    reconnects = []
    for a, b in zip(ordered, ordered[1:]):
        previous = sorted((r for r in frames if r['attempt_id'] == a['attempt_id']), key=lambda r: r['sequence'])
        following = sorted((r for r in frames if r['attempt_id'] == b['attempt_id']), key=lambda r: r['sequence'])
        if previous and following:
            reconnects.append(dict(previous_attempt=a, next_attempt=b,
                last_mux=previous[-1]['encoded_monotonic'], first_mux=following[0]['encoded_monotonic'],
                mux_gap_seconds=following[0]['encoded_monotonic']-previous[-1]['encoded_monotonic']))
    live = [r for r in rows if r['kind'] == 'live_state']
    result = dict(baseline=BASELINE, runtime_checks=checks, attempts=summaries, rtp=rtp, reconnects=reconnects,
        capture=dict(client_start_unix=rows[0]['captured_unix'], client_end_unix=rows[-1]['captured_unix'],
                     http_task_samples=len(tasks), errors=sum(r['kind']=='capture_error' for r in rows)),
        ages={key: distribution([r[key] for r in live if r.get(key) is not None])
              for key in ('oldest_input_age_seconds', 'oldest_lookahead_wait_seconds')},
        limitations=['Pacer due/rebase decisions absent; all sub-100ms non-process/render operations omitted',
                     'Input enqueue/dequeue/backpressure edges absent; queue-full fraction is sampled',
                     'Counterfactual subtraction assumes unchanged work; optimistic bound, not measured improvement',
                     'Stage counters are nested/concurrent; do not sum them',
                     'Receiver subscribes video only; audio integrity and player buffering not observed'])
    (folder/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(attempts={aid: {key: value[key] for key in (
        'window_seconds', 'output_fps', 'nonpacing_ms_per_output', 'queue_full_sample_fraction')}
        for aid, value in summaries.items()}, capture=result['capture']), indent=2))
    if '--assert-realtime' in sys.argv:
        steady = [s for s in summaries.values() if s['window_seconds'] >= 60 and s['progress']]
        assert steady, 'No sufficiently long verified attempt'
        assert all(s['progress']['lag_growth_per_second'] <= .05 for s in steady), \
            'Sustained backlog: media progress falls behind by more than .05s per wall second'


if __name__ == '__main__':
    assert union([(0, 2), (1, 3), (4, 5)]) == 4
    assert distribution([0, 10])['p95'] == 9.5
    assert low_fps([0, .1, 1.1, 2.1], 1)['min'] == 1
    analyze(Path(sys.argv[1]))
