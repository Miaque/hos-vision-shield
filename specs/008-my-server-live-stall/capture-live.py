"""Bounded metadata capture for the two prepared test tasks; never changes tasks."""
import json
from pathlib import Path
import sys
import time
import urllib.request


def capture(root, seconds, api, ids):
    assert 0 < seconds <= 600
    root.mkdir(parents=True, exist_ok=True)
    deadline, written = time.monotonic() + seconds, 0
    with (root / 'live.jsonl').open('w', encoding='utf-8') as target:
        while time.monotonic() < deadline:
            started = time.monotonic()
            for endpoint in ['v1/tasks', *(f'v1/tasks/{id}/diagnostics' for id in ids)]:
                try:
                    with urllib.request.urlopen(f'{api}/{endpoint}', timeout=5) as response:
                        body = response.read(2 * 1024 * 1024 + 1)
                    assert len(body) <= 2 * 1024 * 1024, 'response budget exceeded'
                    value = json.loads(body)
                    if 'attempts' in value:
                        value['attempts_discarded_by_capture'] = max(0, len(value['attempts']) - 4)
                        value['attempts'] = value['attempts'][-4:]
                    if endpoint == 'v1/tasks':
                        for task in value.get('current', []):
                            stream = task.get('stats', {}).get('stream', {})
                            if 'recent_outputs' in stream:
                                stream['recent_outputs'] = stream['recent_outputs'][-64:]
                except Exception as error:
                    value = {'capture_error': type(error).__name__}
                line = json.dumps(dict(endpoint=endpoint, unix=time.time(), monotonic=time.monotonic(),
                                       value=value), separators=(',', ':')) + '\n'
                written += len(line.encode())
                assert written <= 192 * 1024 * 1024, 'capture budget exceeded'
                target.write(line)
            target.flush()
            time.sleep(max(0, min(1 - (time.monotonic() - started), deadline - time.monotonic())))


if __name__ == '__main__':
    capture(Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4:])
