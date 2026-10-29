"""HTTP-only bounded experiment capture, with one frame/operation identity per attempt."""
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 600
TASK = '3c66326a-fd00-4fc8-9e99-9af23fbe45b7'


def main():
    seen_frames, seen_operations, seen_scopes = set(), set(), set()
    seen_runtime = set()
    seen_receipts = set()
    written = 0
    started = time.perf_counter()
    with httpx.Client(base_url='http://127.0.0.1:18000', trust_env=False, timeout=8) as client, \
            (ROOT / 'live.jsonl').open('w', encoding='utf-8') as target:
        def save(kind, **data):
            nonlocal written
            line = json.dumps(dict(kind=kind, captured_unix=time.time(), local_perf=time.perf_counter(), **data),
                              separators=(',', ':')) + '\n'
            written += len(line.encode())
            if written > 96 * 1024 * 1024:
                raise RuntimeError('capture size limit reached')
            target.write(line)
            target.flush()

        for tick in range(DURATION):
            try:
                before = time.perf_counter()
                r = client.get(f'/v1/tasks/{TASK}')
                r.raise_for_status()
                data = r.json()
                stream = data.get('stats', {}).get('stream', {})
                attempt = stream.get('attempt_id')
                for frame in stream.pop('recent_outputs', []):
                    key = (attempt, frame['sequence'])
                    if key not in seen_frames:
                        save('frame', attempt_id=attempt, **frame)
                        seen_frames.add(key)
                save('task', data=data, http_seconds=time.perf_counter() - before,
                     headers={k: v for k, v in r.headers.items() if k.startswith('x-') or k == 'server-timing'})
                r = client.get(f'/v1/tasks/{TASK}/diagnostics')
                r.raise_for_status()
                diag = r.json()
                live = diag.pop('live', None)
                receiver = diag.pop('receiver', None)
                if receiver:
                    rid = receiver['observer_id']
                    for receipt in receiver.pop('frames', []):
                        key = (rid, receipt['sequence'])
                        if key not in seen_receipts:
                            save('internal_frame', observer_id=rid, **receipt)
                            seen_receipts.add(key)
                    save('internal_state', data=receiver)
                    if len(seen_receipts) > 2048:
                        seen_receipts = {key for key in seen_receipts if key[0] == rid and key[1] > receipt['sequence'] - 512}
                save('attempts', data=diag)
                if live:
                    aid = live['attempt_id']
                    if aid not in seen_runtime:
                        save('runtime', attempt_id=aid, data=live['runtime'])
                        seen_runtime.add(aid)
                    for operation in live.get('operations', []):
                        key = (aid, operation['sequence'])
                        if key not in seen_operations:
                            record = dict(operation)
                            record['operation_kind'] = record.pop('kind')
                            save('operation', attempt_id=aid, **record)
                            seen_operations.add(key)
                    for scope in live.get('slow_scopes', []):
                        key = (aid, scope['thread_id'], scope['stage'], scope['ended_monotonic'])
                        if key not in seen_scopes:
                            save('scope', attempt_id=aid, **scope)
                            seen_scopes.add(key)
                    save('live_state', attempt_id=aid, **{k: v for k, v in live.items()
                        if k not in ('attempt_id', 'runtime', 'operations', 'slow_scopes')})
                if tick % 3 == 0:
                    resources = client.get('/v1/diagnostics/resources')
                    save('resources', data=resources.json())
                if tick % 30 == 0:
                    print(json.dumps(dict(seconds=tick, attempt=attempt, frames=len(seen_frames),
                          operations=len(seen_operations), scopes=len(seen_scopes), runtime=len(seen_runtime))), flush=True)
                # Retain enough de-duplication keys for the server's small rolling windows.
                if len(seen_frames) > 4096:
                    seen_frames = {key for key in seen_frames if key[0] == attempt and key[1] > stream['encoded_frames'] - 1024}
                if len(seen_operations) > 4096:
                    seen_operations = {key for key in seen_operations if key[0] == aid and key[1] > operation['sequence'] - 1024}
                if len(seen_scopes) > 4096:
                    seen_scopes = set(sorted(seen_scopes, key=lambda key: key[3])[-2048:])
            except (httpx.HTTPError, ValueError) as error:
                save('capture_error', error_type=type(error).__name__)
            delay = started + tick + 1 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)


if __name__ == '__main__':
    main()
