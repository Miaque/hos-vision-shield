"""Summarize complete RTP frames and disjoint host budgets, preserving evidence gaps."""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {**{f'p{n}': values[min(len(values) - 1, int(len(values) * n / 100))] for n in (50, 95, 99)},
            'max': values[-1], 'min': values[0], 'count': len(values)}


def reception(rows):
    frames = [r for r in rows if r['kind'] == 'frame']
    times = [r.get('last_packet_perf', r.get('monotonic')) for r in frames]
    if len(times) < 2:
        return dict(frames=len(times))
    gaps = [b - a for a, b in zip(times, times[1:])]
    windows = {}
    for seconds in (1, 5):
        starts = [times[0] + index / 10 for index in range(max(0, int((times[-1] - times[0] - seconds) * 10)))]
        counts = [(bisect_right(times, start + seconds) - bisect_right(times, start)) / seconds for start in starts]
        windows[str(seconds)] = distribution(counts)
    return dict(frames=len(frames), seconds=times[-1] - times[0], gaps=distribution(gaps),
                stalls={str(t): sum(x > t for x in gaps) for t in (.2, .5, 1)}, window_fps=windows,
                errors=sum(r['kind'] == 'error' for r in rows))


def analyze(root):
    result = {'rtp': {}, 'wire': {}, 'host': {}, 'same_frame': {}}
    for path in root.glob('*/rtp-*.jsonl'):
        result['rtp'][str(path.relative_to(root))] = reception([json.loads(x) for x in path.open()])
    frames, outputs = {}, {}
    for path in root.glob('*/live.jsonl'):
        group = path.parent.name
        snapshots = {}
        for line in path.open():
            row = json.loads(line)
            if 'diagnostics' not in row['endpoint']:
                continue
            task = row['endpoint'].split('/')[2]
            live = row['value'].get('live', {})
            e0 = live.get('e0')
            if not e0:
                continue
            key = (task, live['attempt_id'])
            snapshots.setdefault(key, []).append(e0)
            for frame in e0['frames']:
                digest = frame.get('encoded_vcl_sha256')
                if digest:
                    frames.setdefault(digest, {}).update(frame)
        for key, samples in snapshots.items():
            a, b = samples[0], samples[-1]
            n = b['fifo_totals'].get('dequeued', 0) - a['fifo_totals'].get('dequeued', 0)
            wall = b['main_wall_seconds'] - a['main_wall_seconds']
            if n <= 0:
                continue
            stages = {k: (v - a['main_exclusive_seconds'].get(k, 0)) * 1000 / n
                      for k, v in b['main_exclusive_seconds'].items()}
            p0, p1 = a['primary_task_totals'], b['primary_task_totals']
            result['host'][f'{group}/{key[0]}/{key[1]}'] = dict(frames=n, wall=wall,
                main_ms_per_frame=dict(sorted(stages.items(), key=lambda x: -x[1])),
                future_wait_ms_per_frame=(p1.get('wait_seconds', 0) - p0.get('wait_seconds', 0)) * 1000 / n,
                future_wait_plate_ms_per_frame=(p1.get('wait_stage:forward-primary-plate_seconds', 0)
                                               - p0.get('wait_stage:forward-primary-plate_seconds', 0)) * 1000 / n)
    for path in (root / 'wire-baseline').glob('rtp-output*.jsonl'):
        for line in path.open():
            row = json.loads(line)
            if row['kind'] == 'frame':
                outputs[row['vcl_sha256']] = row
    for path in root.glob('wire-fixed-*.jsonl'):
        rows = [json.loads(x) for x in path.open()]
        result['wire'][path.name] = dict(reception(rows), counts=dict(Counter(r['kind'] for r in rows)))
        matched, timeline = [], []
        for wire in rows:
            if wire['kind'] != 'frame':
                continue
            digest = wire['vcl_sha256']
            frame, receiver = frames.get(digest, {}), outputs.get(digest)
            if 'mux_completed_monotonic' not in frame or receiver is None:
                continue
            match = dict(frame_id=frame['frame_id'], digest=digest, source_clock=frame.get('raw_frame_clock'),
                received=frame['received_monotonic'], decoded=frame['decoded_monotonic'],
                processing=frame['processing_started_monotonic'],
                lookahead_enter=frame['lookahead_entered_monotonic'],
                lookahead_release=frame['lookahead_released_monotonic'], render=frame['render_completed_monotonic'],
                encode=frame['encode_completed_monotonic'], mux=frame['mux_completed_monotonic'],
                tcp_egress=wire['monotonic'], rtp_receive=receiver['last_packet_perf'],
                incomplete=frame.get('incomplete_future_window'))
            matched.append(match)
        for a, b in zip(matched, matched[1:]):
            gap = b['rtp_receive'] - a['rtp_receive']
            if gap > .2:
                timeline.append(dict(before=a, after=b, receiver_gap=gap,
                                     mux_gap=b['mux'] - a['mux'], egress_gap=b['tcp_egress'] - a['tcp_egress']))
        result['same_frame'][path.name] = dict(matches=len(matched),
            mux_to_egress=distribution([r['tcp_egress'] - r['mux'] for r in matched]),
            egress_to_receiver=distribution([r['rtp_receive'] - r['tcp_egress'] for r in matched]),
            stalls=sorted(timeline, key=lambda x: -x['receiver_gap'])[:12])
    (root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    result = analyze(Path(sys.argv[1]))
    print(json.dumps({k: v for k, v in result.items() if k in ('rtp', 'host')}, indent=2))
