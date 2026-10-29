"""Trace coverage, FIFO/lookahead ages and lifecycle gaps supplement cumulative budgets."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys

from analyze import distribution


def summarize(rows):
    frames, pacer, audio, outputs = {}, {}, {}, {}
    states, events, attempts = defaultdict(list), defaultdict(list), {}
    for row in rows:
        kind, data, aid = row['kind'], row.get('data', {}), row.get('attempt_id')
        if kind == 'frame':
            key = (aid, data['frame_id'])
            if data['revision'] > frames.get(key, {}).get('revision', 0):
                frames[key] = data
        elif kind in ('pacer', 'audio_clock', 'audio_output'):
            target = dict(pacer=pacer, audio_clock=audio, audio_output=outputs)[kind]
            target[(aid, data['sequence'])] = data
        elif kind == 'e0_state':
            states[aid].append(data)
        elif kind == 'event':
            events[aid].append(data)
        elif kind == 'attempt' and 'attempt_id' in data:
            attempts[data['attempt_id']] = data
    result = {}
    for aid, group in states.items():
        lo, hi = group[0]['sampled_monotonic'], group[-1]['sampled_monotonic']
        last = attempts.get(aid, {}).get('e0')
        if last is not None:
            hi = max(hi, last['ended_monotonic'])
        selected = [f for (a, _), f in frames.items() if a == aid]
        published = [f for f in selected if lo <= f.get('mux_completed_monotonic', -1) <= hi]
        pairs = dict(fifo_wait=('enqueue_monotonic', 'dequeue_monotonic'),
                     lookahead_residence=('lookahead_entered_monotonic', 'lookahead_released_monotonic'),
                     release_to_render=('lookahead_released_monotonic', 'render_started_monotonic'),
                     render_wall=('render_started_monotonic', 'render_completed_monotonic'),
                     final_render_to_mux=('render_completed_monotonic', 'mux_completed_monotonic'))
        durations = {name: distribution([f[end]-f[start] for f in selected
                     if start in f and end in f and lo <= f[end] <= hi]) for name, (start, end) in pairs.items()}
        release_batches = Counter(round(f['lookahead_released_monotonic'], 2) for f in selected
                                  if 'lookahead_released_monotonic' in f)
        ordered_pacer = sorted((r for (a, _), r in pacer.items() if a == aid), key=lambda r:r['sequence'])
        ordered_audio = sorted((r for (a, _), r in audio.items() if a == aid), key=lambda r:r['sequence'])
        def value(clock, field):
            return None if clock.get(field) is None or clock.get('time_base') is None else clock[field]*Fraction(clock['time_base'])
        regressions = []
        for a, b in zip(ordered_audio, ordered_audio[1:]):
            if b['sequence'] != a['sequence']+1:
                continue
            before, after = value(a['raw'], 'dts'), value(b['raw'], 'dts')
            if before is not None and after is not None and after <= before:
                regressions.append(dict(previous=a, current=b))
        evidence = [dict(sequence=p['sequence'], frame_id=p.get('frame_id'), moment=p['moment'],
            raw=p.get('raw_frame_clock'), normalized=p.get('normalized_clock'), encoder=p.get('encoder_packet_clock'),
            decision_at=p['decision_at'], due=p['due'], actual=p['actual'], remaining=p['remaining'],
            conditional_no_rebase_due=p['conditional_no_rebase_due'], rebase_reason=p['rebase_reason'],
            actual_wait_seconds=p['actual_wait_seconds'], conditional_rebase_request_seconds=p['conditional_rebase_request_seconds'],
            retained_anti_burst_request_seconds=p['retained_anti_burst_request_seconds'])
            for p in ordered_pacer if p['conditional_rebase_request_seconds'] > .01 and p['remaining'] > 0][:5]
        result[aid] = dict(retained_frames=len(selected), retained_published_frames=len(published),
            duration_seconds=durations, sampled_release_reasons=dict(Counter(f.get('release_reason') for f in selected)),
            approximate_largest_release_group_in_10ms=max(release_batches.values(), default=0),
            event_boundaries=events[aid], worker_boundaries=attempts.get(aid, {}).get('e0_boundaries'),
            retained_audio_regression_examples=regressions[:3],
            retained_audio_count=len(ordered_audio),
            retained_audio_output_count=sum(a == aid for a, _ in outputs),
            conditional_rebase_examples=evidence)
    ordered = sorted((row for row in attempts.values() if row.get('e0')),
                     key=lambda r:r['started_monotonic'])
    reconnects = []
    for previous in ordered:
        following = min((row for row in attempts.values() if row['started_monotonic'] > previous['started_monotonic']),
                        key=lambda r:r['started_monotonic'], default=None)
        if following is None:
            continue
        previous_last = previous['e0'].get('last_output')
        following_state = states.get(following['attempt_id'], [{}])[-1]
        following_first = (following.get('e0') or following_state).get('first_output')
        if previous_last and following_first:
            reconnects.append(dict(previous_attempt=previous['attempt_id'], next_attempt=following['attempt_id'],
                reason=previous.get('reason'), mux_gap_seconds=following_first['monotonic']-previous_last['monotonic'],
                previous_ended_monotonic=previous['e0']['ended_monotonic'],
                next_worker_started_monotonic=following['started_monotonic'],
                last_mux=previous_last['monotonic'], next_first_mux=following_first['monotonic']))
    return dict(attempts=result, reconnects=reconnects,
        note='Frame distributions and examples use retained samples only; global costs use e0-summary cumulative counters')


def self_check():
    common = dict(frame_id=1, enqueue_monotonic=1, dequeue_monotonic=2)
    rows = [dict(kind='e0_state', attempt_id='a', data=dict(sampled_monotonic=0)),
            dict(kind='frame', attempt_id='a', data=dict(common, revision=2)),
            dict(kind='frame', attempt_id='a', data=dict(common, revision=1, dequeue_monotonic=99)),
            dict(kind='e0_state', attempt_id='a', data=dict(sampled_monotonic=3))]
    result = summarize(rows)
    assert result['attempts']['a']['duration_seconds']['fifo_wait']['mean'] == 1
    print('E0 detail identity/coverage self-check passed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        path = Path(sys.argv[1])
        result = summarize([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()])
        path.with_name('e0-details.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(dict(attempts=len(result['attempts']), reconnects=result['reconnects'])))
