"""Join bounded host and wire evidence by access-unit hashes, never by average FPS."""
from collections import Counter, defaultdict
from bisect import bisect_right
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location('analysis', Path(__file__).with_name('analyze.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rows(path):
    if path.exists():
        boundary = path.parent / 'end.json'
        end = json.loads(boundary.read_text())['unix'] if boundary.exists() else float('inf')
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('unix', 0) <= end:
                    yield row


def clocks(values):
    reports = defaultdict(list)
    for row in values:
        if row['kind'] == 'sr':
            reports[(row['connection'], row['ssrc'], row['channel'])].append(row)
    result = []
    for key, samples in reports.items():
        rate = 90000 if key[2] == 1 else 8000
        residual = []
        for before, after in zip(samples, samples[1:]):
            delta = (after['timestamp'] - before['timestamp'] + 2**31) % 2**32 - 2**31
            residual.append((after['ntp'] - before['ntp']) - delta / rate)
        result.append(dict(connection=key[0], ssrc=key[1], channel=key[2], reports=len(samples),
                           affine_residual_seconds=analysis.distribution([abs(x) for x in residual])))
    return result


def reception_window(values, begin, end):
    values = [v for v in values if begin <= v['unix'] <= end]
    result = analysis.reception(values)
    times = [v['unix'] for v in values if v['kind'] == 'frame']
    result.update(observation_seconds=end-begin,
                  first_frame_wait_seconds=times[0]-begin if times else end-begin,
                  tail_without_picture_seconds=end-times[-1] if times else end-begin)
    result['window_fps'] = {}
    for seconds in (1, 5):
        starts = [begin + i/10 for i in range(max(0, int((end-begin-seconds)*10)))]
        fps = [(bisect_right(times, t+seconds)-bisect_right(times, t))/seconds for t in starts]
        result['window_fps'][str(seconds)] = analysis.distribution(fps)
    return result


def main(root, group='final-diagnostic', task_id=None):
    result = dict(reception={}, same_frame={}, rtcp={}, formal_phases={}, main_cost={})
    for name in sorted({path.parent.name for path in root.glob('*/raw.jsonl')}):
        for role in ('raw', 'output'):
            path = root / name / (role + '.jsonl')
            if not path.exists():
                continue
            values = list(rows(path))
            result['reception'][f'{name}/{role}'] = dict(analysis.reception(values),
                counts=dict(Counter(v['kind'] for v in values)))
            result['rtcp'][f'{name}/{role}'] = clocks(values)
            if name.startswith('serial-formal-'):
                start = json.loads((path.parent/'start.json').read_text())
                result['formal_phases'][f'{name}/{role}'] = reception_window(
                    values, start.get('rtp_started_unix', start['unix']), start['unix'] + start['seconds'])
            if name == 'formal-final':
                begin = json.loads((path.parent/'start.json').read_text())['unix']
                transition = json.loads((path.parent/'dual-start.json').read_text())['unix']
                for phase, lo, hi in [('single', begin, transition), ('dual', transition, begin+1800)]:
                    result['formal_phases'][f'{phase}/{role}'] = reception_window(values, lo, hi)
    directory = root / group
    frames, packets, boundaries, snapshots = {}, {}, {}, {}
    for row in rows(directory / 'live.jsonl'):
        if task_id and row['endpoint'] != f'v1/tasks/{task_id}/diagnostics':
            continue
        live = row.get('value', {}).get('live', {})
        e0 = live.get('e0', {})
        attempt = live.get('attempt_id')
        if not e0:
            continue
        snapshots.setdefault((row['endpoint'], attempt), []).append(e0)
        boundaries[attempt] = e0.get('events', [])
        for value in e0.get('video_clocks', []):
            clock = value['raw']
            packets[(attempt, clock.get('pts'), clock.get('time_base'))] = value
        for frame in e0.get('frames', []):
            key = attempt, frame['frame_id']
            if frame.get('revision', 0) >= frames.get(key, {}).get('revision', 0):
                frames[key] = frame
    for (endpoint, attempt), samples in snapshots.items():
        before, after = samples[0], samples[-1]
        count = after['fifo_totals'].get('dequeued', 0) - before['fifo_totals'].get('dequeued', 0)
        if count > 0:
            stages = {key: (value-before['main_exclusive_seconds'].get(key, 0))*1000/count
                      for key, value in after['main_exclusive_seconds'].items()}
            result['main_cost'][endpoint + '/' + attempt] = dict(frames=count,
                wall_ms_per_frame=(after['main_wall_seconds']-before['main_wall_seconds'])*1000/count,
                exclusive_ms_per_frame=dict(sorted(stages.items(), key=lambda item: -item[1])))
    raw, output, wire = defaultdict(list), defaultdict(list), defaultdict(list)
    for value in rows(directory / 'raw.jsonl'):
        if value['kind'] == 'frame':
            raw[value['vcl_sha256']].append(value)
    receiver_index = 0
    for value in rows(directory / 'output.jsonl'):
        if value['kind'] == 'frame':
            value['receiver_index'] = receiver_index
            receiver_index += 1
            output[value['vcl_sha256']].append(value)
    for path in directory.glob('wire-*.jsonl'):
        for value in rows(path):
            if value['kind'] == 'frame':
                wire[value['vcl_sha256']].append(value)
    offsets = defaultdict(list)
    for (attempt, _), frame in frames.items():
        clock = frame['raw_frame_clock']
        packet = packets.get((attempt, clock.get('pts'), clock.get('time_base')), {})
        if clock.get('pts') is None:
            continue
        stamp = int(clock['pts'] * Fraction(clock['time_base']) * 90000)
        for value in raw.get(packet.get('vcl_sha256'), []):
            lag = frame['received_monotonic'] - value['monotonic']
            if lag >= -.02:
                offsets[(attempt, value['connection'], (value['timestamp'] - stamp) % 2**32)].append(lag)
    selected = {}
    for attempt, _, _ in offsets:
        choices = [(key, lags) for key, lags in offsets.items() if key[0] == attempt]
        selected[attempt] = max(choices, key=lambda item: (len(item[1]), -sum(item[1])/len(item[1])))[0]
    host_hashes = Counter(frame.get('encoded_vcl_sha256') for frame in frames.values())
    joined = []
    for (attempt, fid), frame in frames.items():
        digest, mux = frame.get('encoded_vcl_sha256'), frame.get('mux_completed_monotonic')
        if not digest or mux is None or digest not in wire or digest not in output:
            continue
        if host_hashes[digest] != 1 or len(wire[digest]) != 1 or len(output[digest]) != 1:
            continue
        sent, received = wire[digest][0], output[digest][0]
        assert sent['timestamp'] == received['timestamp'], 'Wire/receiver RTP timestamps differ'
        point = dict(attempt=attempt, digest=digest, tcp_egress=sent['monotonic'],
                     rtp_receive=received['monotonic'], receiver_index=received['receiver_index'], **frame)
        clock = frame['raw_frame_clock']
        packet = packets.get((attempt, clock.get('pts'), clock.get('time_base')), {})
        stamp = int(clock['pts'] * Fraction(clock['time_base']) * 90000)
        offset = selected.get(attempt)
        source = [v for v in raw.get(packet.get('vcl_sha256'), []) if offset and
                  v['connection'] == offset[1] and
                  abs((v['timestamp'] - stamp - offset[2] + 2**31) % 2**32 - 2**31) <= 1]
        # VCL hash plus an RTP/PTS offset (one rescaling tick tolerance) identify the replay cycle.
        if len(source) == 1:
            point.update(source_rtp_receive=source[0]['monotonic'], source_digest=packet['vcl_sha256'])
        joined.append(point)
    joined.sort(key=lambda v: v['rtp_receive'])
    pairs = [(a, b) for a, b in zip(joined, joined[1:]) if a['attempt'] == b['attempt']
             and b['receiver_index'] == a['receiver_index'] + 1]
    source_matched = [v for v in joined if 'source_rtp_receive' in v]
    result['same_frame'] = dict(matches=len(joined), source_matches=len(source_matched),
        source_rtp_pts_offsets=[dict(attempt=k[0], connection=k[1], offset=k[2], support=len(v), selected=selected.get(k[0])==k)
                                for k,v in sorted(offsets.items(), key=lambda item: -len(item[1]))[:12]],
        mux_to_egress=analysis.distribution([v['tcp_egress'] - v['mux_completed_monotonic'] for v in joined]),
        mux_start_to_egress=analysis.distribution([v['tcp_egress'] - v['mux_started_monotonic'] for v in joined]),
        egress_to_receive=analysis.distribution([v['rtp_receive'] - v['tcp_egress'] for v in joined]),
        source_receive_to_decode=analysis.distribution([v['decoded_monotonic'] - v['source_rtp_receive'] for v in source_matched]),
        fifo_wait=analysis.distribution([v['dequeue_monotonic'] - v['enqueue_monotonic'] for v in joined]),
        backpressure=analysis.distribution([v.get('backpressure_seconds', 0) for v in joined]),
        source_matched_timeline=source_matched[::max(1, len(source_matched)//12)],
        stalls=[dict(before=a, after=b, gap=b['rtp_receive']-a['rtp_receive'])
                for a, b in sorted(pairs, key=lambda p: p[1]['rtp_receive']-p[0]['rtp_receive'], reverse=True)[:12]],
        lifecycle=boundaries)
    (root / (group + '-summary.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(reception=result['reception'], same_frame={k:v for k,v in result['same_frame'].items()
                      if k not in ('stalls', 'lifecycle', 'source_matched_timeline')}), indent=2))


if __name__ == '__main__':
    assert analysis.reception([dict(kind='frame', monotonic=t) for t in (0, .04, 2, 2.04, 7)])['stalls']['1'] == 2
    assert clocks([dict(kind='sr', connection=1, ssrc=1, channel=1, timestamp=i*90000, ntp=100+i)
                   for i in (0, 5)])[0]['affine_residual_seconds']['max'] == 0
    assert reception_window([], 0, 10)['window_fps']['1']['max'] == 0
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'final-diagnostic',
         sys.argv[3] if len(sys.argv) > 3 else None)
