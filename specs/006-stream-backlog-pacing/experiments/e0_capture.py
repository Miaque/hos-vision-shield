"""HTTP GET only. Bounded, de-duplicated E0 capture; never deploys or changes a task."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx

from e0_analyze import verify_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--task', default='27c39007-0aee-4185-b8fa-9d1ece4de058')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--base-url', default='http://127.0.0.1:18000')
    args = parser.parse_args()
    assert 1 <= args.seconds <= 1800
    args.directory.mkdir(parents=True, exist_ok=True)
    started, written, last_print = time.monotonic(), 0, 0
    runtimes, revisions, sequences, attempts = set(), {}, {}, {}
    previous, failed = {}, 0
    last_rtp = started
    clock_identity = None
    names = dict(frames='frame', pacer='pacer', audio_clocks='audio_clock', video_clocks='video_clock',
                 audio_outputs='audio_output', events='event')
    with httpx.Client(base_url=args.base_url, trust_env=False, timeout=8) as client, \
            (args.directory/'e0.jsonl').open('x', encoding='utf-8') as stream:
        def save(kind, **data):
            nonlocal written
            line = json.dumps(dict(kind=kind, captured_unix=time.time(), **data), separators=(',', ':'))+'\n'
            written += len(line.encode())
            if written > 96*1024*1024:
                raise RuntimeError('Capture byte limit; service unchanged')
            stream.write(line)
            stream.flush()
        task = client.get('/v1/tasks/'+args.task)
        task.raise_for_status()
        assert task.json()['state'] == 'running' and task.json()['ready']
        while time.monotonic()-started < args.seconds:
            tick = time.monotonic()
            try:
                response = client.get('/v1/tasks/'+args.task+'/diagnostics')
                response.raise_for_status()
                data = response.json()
                live, receiver = data['live'], data.get('receiver')
                aid, e0 = live['attempt_id'], live['e0']
                if aid not in runtimes:
                    verify_runtime(live['runtime'], args.ref)
                    current_clock = live['runtime'].get('clock_identity')
                    if clock_identity is None:
                        clock_identity = current_clock
                    assert clock_identity == current_clock, 'Server clock identity changed'
                    save('runtime', attempt_id=aid, data=live['runtime'])
                    runtimes.add(aid)
                    revisions.clear()
                assert e0['schema_version'] == 1
                decisions = e0['pacer_totals'].get('decisions', 0)
                assert decisions >= previous.get(aid, 0), 'Counter went backwards within one attempt'
                previous[aid] = decisions
                for key, kind in names.items():
                    for row in e0.pop(key):
                        if key == 'frames':
                            fid = row['frame_id']
                            if row['revision'] <= revisions.get(fid, 0):
                                continue
                            revisions[fid] = row['revision']
                        else:
                            skey = (aid, key)
                            if row['sequence'] <= sequences.get(skey, 0):
                                continue
                            sequences[skey] = row['sequence']
                        save(kind, attempt_id=aid, data=row)
                revisions = {fid: rev for fid, rev in revisions.items() if fid > e0['totals']['frames']-256}
                save('e0_state', attempt_id=aid, data=e0)
                for attempt in data.get('attempts', []):
                    key = attempt['run_id']
                    stamp = hashlib.sha256(json.dumps(attempt, sort_keys=True).encode()).hexdigest()
                    if attempts.get(key) != stamp:
                        save('attempt', data=attempt)
                        attempts[key] = stamp
                if receiver:
                    assert receiver['clock'] == live['runtime']['clock_identity'], 'Different server clocks'
                    skey = (receiver['observer_id'], 'rtp')
                    for frame in receiver.pop('frames'):
                        if frame['sequence'] > sequences.get(skey, 0):
                            save('rtp_frame', observer_id=receiver['observer_id'], data=frame)
                            sequences[skey] = frame['sequence']
                            last_rtp = time.monotonic()
                    save('receiver_state', data=receiver)
                assert time.monotonic()-last_rtp < 15, 'No fresh RTP observations for 15s'
                failed = 0
            except httpx.HTTPError as error:
                failed += 1
                save('http_error', error_type=type(error).__name__)
                if failed >= 3:
                    raise RuntimeError('Three HTTP failures; service unchanged') from None
            if time.monotonic()-last_print >= 30:
                print(json.dumps(dict(seconds=round(time.monotonic()-started, 1), bytes=written)), flush=True)
                last_print = time.monotonic()
            time.sleep(max(0, 1-(time.monotonic()-tick)))
    print('Capture finished; no task or deployment was changed')


if __name__ == '__main__':
    main()
