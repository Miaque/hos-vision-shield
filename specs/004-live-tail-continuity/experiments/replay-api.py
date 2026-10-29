"""Fixed-clip P0 experiment through the authorized service API only."""

import hashlib
import json
from pathlib import Path
import sys
import time

import av
import httpx


source, directory, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert mode in ('normal', 'host', 'local-face')
directory.mkdir(parents=True, exist_ok=False)


def save(name, value):
    (directory / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), 'utf-8')


def media(path):
    with av.open(str(path)) as reader:
        video = [(str(frame.pts * frame.time_base), frame.width, frame.height)
                 for frame in reader.decode(video=0)]
    with av.open(str(path)) as reader:
        audio = [(str(packet.pts * packet.time_base), str(packet.dts * packet.time_base),
                  str(packet.duration * packet.time_base), hashlib.sha256(bytes(packet)).hexdigest())
                 for packet in reader.demux(audio=0) if packet.dts is not None] if reader.streams.audio else []
    return video, audio


headers = {'X-Detection-Interval': '3', 'X-Tracking-Mode': 'flow',
           'Idempotency-Key': 'local-face-p0-' + directory.name}
if mode != 'normal':
    headers['X-Detection-Profile'] = mode
save('request.json', dict(mode=mode, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          source=str(source.resolve()), headers=headers, started_unix=time.time(),
                          scope='File diagnostic with existing live tasks; not exclusive-GPU acceptance.'))
with httpx.Client(base_url='http://127.0.0.1:18000', trust_env=False, timeout=180) as client:
    client.get('/health/ready').raise_for_status()
    with source.open('rb') as handle:
        response = client.post('/v1/file-tasks/uploads', headers=headers,
                               data={'kind': 'video', 'redaction_mode': 'solid'},
                               files={'file': (source.name, handle, 'video/mp4')})
    response.raise_for_status()
    task_id = response.json()['id']
    save('task-id.json', dict(task_id=task_id))
    print(json.dumps(dict(event='submitted', task_id=task_id, mode=mode)), flush=True)
    started, next_progress = time.monotonic(), 0
    with (directory / 'observations.jsonl').open('x', encoding='utf-8') as log:
        while True:
            assert time.monotonic() - started < 600, 'Diagnostic exceeded 10-minute budget'
            response = client.get('/v1/tasks/' + task_id)
            response.raise_for_status()
            task = response.json()
            timing = {key: value for key, value in response.headers.items()
                      if key in ('server-timing', 'x-detection-counts', 'x-video-decoder', 'x-video-encoder')}
            log.write(json.dumps(dict(unix=time.time(), task=task, headers=timing)) + '\n')
            log.flush()
            if time.monotonic() >= next_progress:
                print(json.dumps(dict(event='progress', state=task['state'], stats=task['stats'])), flush=True)
                next_progress = time.monotonic() + 30
            if task['state'] in ('succeeded', 'failed', 'cancelled'):
                save('task.json', dict(task=task, headers=timing, wall_seconds=time.monotonic() - started))
                assert task['state'] == 'succeeded', 'Diagnostic failed; see task.json'
                assert task['stats']['dropped_frames'] == 0
                break
            time.sleep(2)
    if mode != 'normal':
        response = client.get('/v1/tasks/' + task_id + '/diagnostics')
        response.raise_for_status()
        report = response.json()
        save('diagnostics.json', report)
        for sample in report['profile'].get('device_samples', []):
            response = client.get('/v1/tasks/' + task_id + '/diagnostics', params={'sample': sample['sample']})
            response.raise_for_status()
            (directory / sample['trace']).write_bytes(response.content)
    target = directory / 'redacted.mp4'
    with client.stream('GET', '/v1/tasks/' + task_id + '/result') as response, target.open('xb') as handle:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            handle.write(chunk)
    original, output = media(source), media(target)
    assert original == output, 'Frame/dimension/PTS/audio integrity changed'
    assert len(output[0]) == task['stats']['processed_frames']
    save('integrity.json', dict(frames=len(output[0]), audio_packets=len(output[1]),
                                same_dimensions_pts_audio=True,
                                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    print(json.dumps(dict(event='complete', task_id=task_id, stats=task['stats'],
                          frames=len(output[0]), same_dimensions_pts_audio=True)), flush=True)
