"""Analyze cumulative E0 budgets; retained samples only establish trace coverage."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from vision_shield.media.stream_e0 import runtime_fingerprint
from analyze import distribution, low_fps

REQUIRED = {'media.stream_e0', 'media.stream_pacing', 'media.streams', 'media.plate_lookahead',
            'detection.timing', 'detection.quality', 'jobs.tasks', 'config'}


def transported_fingerprint(runtime):
    # Redis JSON transport writes integral float settings (16.0) as integers (16).
    # Restore the three declared PositiveSeconds fields before checking the worker's digest.
    effective = dict(runtime.get('effective_config') or {})
    for key in ('audio_buffer_seconds', 'rtsp_connect_timeout', 'rtsp_read_timeout'):
        if key in effective:
            effective[key] = float(effective[key])
    restored = dict(runtime, effective_config=effective) if 'effective_config' in runtime else runtime
    return runtime_fingerprint(restored)


def verify_runtime(runtime, ref):
    hashes = runtime['source_sha256']
    assert len(hashes) == 20 and REQUIRED <= hashes.keys(), 'Missing E0 source provenance'
    for name, expected in hashes.items():
        blob = subprocess.check_output(['git', 'show', f'{ref}:src/vision_shield/{name.replace(".", "/")}.py'])
        assert hashlib.sha256(blob).hexdigest() == expected, f'Runtime mismatch: {name}'
    assert runtime['output_mux_options'] == {'rtsp_transport': 'tcp', 'max_interleave_delta': '100000'}
    assert runtime['effective_config']['stream_e0_diagnostics'] is True
    for key, value in dict(stream_output_pacing=True, frame_buffer_size=2, detection_interval=3,
                            tracking='flow', redaction_mode='solid', audio_enabled=True).items():
        assert runtime['effective_config'][key] == value, f'Configuration changed: {key}'


def difference(after, before):
    return {key: value-before.get(key, 0) for key, value in after.items()}


def budget(last, first=None):
    first = first or {}
    totals = difference(last['pacer_totals'], first.get('pacer_totals', {}))
    partitions = ('conditional_rebase_wait_seconds', 'retained_anti_burst_wait_seconds',
                  'retained_media_wait_seconds', 'timer_overrun_seconds')
    assert abs(sum(totals.get(k, 0) for k in partitions)-totals.get('actual_wait_seconds', 0)) < 1e-6
    exclusive = difference(last['main_exclusive_seconds'], first.get('main_exclusive_seconds', {}))
    wall = last['main_wall_seconds']-first.get('main_wall_seconds', 0)
    residual = last['main_unattributed_seconds']-first.get('main_unattributed_seconds', 0)
    assert abs(sum(exclusive.values())+residual-wall) < 1e-5
    categories = defaultdict(float)
    for name, seconds in exclusive.items():
        category = ('pacer_wait' if name == 'wait:pacer' else
            'input_condition_wait' if name == 'wait:input-condition' else
            'explicit_sync_or_future_wait' if name in ('stage:explicit-sync', 'stage:wait-primary') else
            'video_encode' if name in ('op:video-encode', 'op:encoder-flush') else
            'mux' if name in ('op:video-mux', 'op:audio-mux') else
            'pacer_rule_and_probe_wall' if name == 'op:output-pacing' else
            'audio_codec_and_clock_work' if name == 'op:audio-process' else
            'report' if name == 'op:report' else
            'reader_join' if name == 'op:reader-join' else 'other_host_work')
        categories[category] += seconds
    categories['unattributed'] += residual
    clock = {key: value for key, value in difference(last['clock_totals'], first.get('clock_totals', {})).items()
             if not key.endswith('_max_forward_step_seconds')}
    fifo = difference(last['fifo_totals'], first.get('fifo_totals', {}))
    fifo['capacity'] = last['fifo_totals'].get('capacity')
    fifo['backpressure_including_active_seconds'] = (fifo.get('backpressure_seconds', 0)
        + last.get('active_backpressure_seconds', 0)-first.get('active_backpressure_seconds', 0))
    fifo['full_occupancy_including_active_seconds'] = (fifo.get('full_occupancy_seconds', 0)
        + last.get('active_full_occupancy_seconds', 0)-first.get('active_full_occupancy_seconds', 0))
    packets = clock.get('video_mux_packets', 0)
    non_pacing = wall-totals.get('actual_wait_seconds', 0)
    a = first.get('last_output') if first else last.get('first_output')
    b = last.get('last_output')
    progress = None
    if a and b and b['monotonic'] > a['monotonic']:
        elapsed = b['monotonic']-a['monotonic']
        progress = {key: dict(media_seconds=b[key]-a[key], wall_seconds=elapsed,
            lag_growth_seconds=elapsed-(b[key]-a[key]), lag_growth_per_second=1-(b[key]-a[key])/elapsed)
            for key in ('source_seconds', 'normalized_seconds') if a.get(key) is not None and b.get(key) is not None}
    return dict(wall_seconds=wall, pacer=totals, main_exclusive_seconds=exclusive,
        categories_seconds=dict(categories), fifo=fifo,
        clock=clock, worker_mux_packets=packets,
        clock_maxima_lifetime={key: value for key, value in last['clock_totals'].items()
                               if key.endswith('_max_forward_step_seconds')},
        non_pacer_wall_ms_per_mux_packet=non_pacing*1000/packets if packets else None,
        media_progress=progress,
        limitation='Conditional rebase contribution assumes observed arrivals and prior actual releases; not an E1 throughput prediction')


def analyze(path, ref):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    identities = set()
    for row in rows:
        if row['kind'] == 'runtime':
            verify_runtime(row['data'], ref)
            identities.add(transported_fingerprint(row['data']))
    groups, ends = defaultdict(list), {}
    for row in rows:
        if row['kind'] == 'e0_state':
            groups[row['attempt_id']].append(row['data'])
        elif row['kind'] == 'attempt' and row['data'].get('e0'):
            ends[row['data']['attempt_id']] = row['data']['e0']
    assert groups, 'No E0 cumulative evidence'
    t0 = min(group[0]['sampled_monotonic'] for group in groups.values())
    t1 = max(group[-1]['sampled_monotonic'] for group in groups.values())
    for aid, final in ends.items():
        if aid not in groups and t0 <= final['started_monotonic'] <= final['ended_monotonic'] <= t1:
            groups[aid] = [final]
    results = {}
    for aid, group in groups.items():
        first, last = group[0], group[-1]
        if aid in ends and ends[aid]['ended_monotonic'] <= t1:
            last = ends[aid]
        assert first['runtime_fingerprint'] in identities and last['runtime_fingerprint'] in identities
        before = None if first['started_monotonic'] >= t0 else first
        result = budget(last, before)
        # Normal source-PTS windows exclude observed worker clock repairs and large gaps.
        # They do not prove that the upstream network delivers continuously.
        normal = []
        for start, end in zip(group, group[1:]):
            counters = ('video_raw_frame_pts_forward_jumps_over_500ms', 'video_raw_frame_pts_nonincreasing')
            a, b = start.get('last_output'), end.get('last_output')
            if (a and b and b['monotonic'] > a['monotonic'] and
                    all(start['clock_totals'].get(key, 0) == end['clock_totals'].get(key, 0) for key in counters)):
                seconds = b['monotonic']-a['monotonic']
                media = b['source_seconds']-a['source_seconds']
                if seconds >= 1:
                    normal.append(dict(wall_seconds=seconds, media_seconds=media,
                        lag_growth_seconds=seconds-media, lag_growth_per_second=1-media/seconds))
        result['no_observed_large_pts_jump_windows'] = dict(n=len(normal),
            wall_seconds=sum(x['wall_seconds'] for x in normal),
            media_seconds=sum(x['media_seconds'] for x in normal),
            lag_growth_per_second=(sum(x['lag_growth_seconds'] for x in normal)/sum(x['wall_seconds'] for x in normal)
                                   if normal else None),
            note='Excludes observed >500ms source PTS jumps/nonincreasing PTS; smaller gaps and source arrival jitter may remain')
        lower = (before or {}).get('pacer_totals', {}).get('decisions', 0)
        observed = {r['data']['sequence'] for r in rows if r['kind']=='pacer' and r['attempt_id']==aid
                    and lower < r['data']['sequence'] <= last['pacer_totals'].get('decisions', 0)}
        result.update(pacer_records=len(observed), pacer_records_missing=result['pacer'].get('decisions', 0)-len(observed),
                      ended=last.get('ended_monotonic') is not None)
        results[aid] = result
    output = dict(ref=ref, attempts=results,
        note='Overall wait costs come from cumulative counters, never from an incomplete frame ring')
    received = [r for r in rows if r['kind']=='rtp_frame' and t0 <= r['data']['received_monotonic'] <= t1]
    gaps = [b['data']['received_monotonic']-a['data']['received_monotonic'] for a, b in zip(received, received[1:])
            if (a['observer_id'], a['data']['connection']) == (b['observer_id'], b['data']['connection'])
            and b['data']['sequence'] == a['data']['sequence']+1]
    times = [r['data']['received_monotonic'] for r in received]
    output['rtp'] = dict(frames=len(received), connection_interval_seconds=distribution(gaps),
        over_500ms=sum(x > .5 for x in gaps), over_1000ms=sum(x > 1 for x in gaps),
        effective_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
        low_1s_fps=low_fps(times, 1), low_5s_fps=low_fps(times, 5),
        receipt_sequence_gaps=sum(a['observer_id']==b['observer_id'] and
            b['data']['sequence'] != a['data']['sequence']+1 for a,b in zip(received, received[1:])),
        note='Receipt sequence is not the network RTP packet sequence; no claim of zero packet loss')
    path.with_name('e0-summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


def self_check():
    first = dict(pacer_totals={'decisions': 100, 'actual_wait_seconds': 1, 'retained_media_wait_seconds': 1},
                 main_exclusive_seconds={'wait:pacer': 1}, main_wall_seconds=2, main_unattributed_seconds=1)
    last = dict(pacer_totals={'decisions': 2100, 'actual_wait_seconds': 3, 'retained_media_wait_seconds': 2,
        'conditional_rebase_wait_seconds': 1}, main_exclusive_seconds={'wait:pacer': 3, 'stage:track-recheck': 2},
        main_wall_seconds=7, main_unattributed_seconds=2, clock_totals={'video_mux_packets': 2000}, fifo_totals={})
    result = budget(last, first)
    assert result['pacer']['decisions'] == 2000 and result['pacer']['conditional_rebase_wait_seconds'] == 1
    assert sum(result['categories_seconds'].values()) == 5
    print('E0 cumulative-budget self-check passed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        analyze(Path(sys.argv[1]), sys.argv[2])
