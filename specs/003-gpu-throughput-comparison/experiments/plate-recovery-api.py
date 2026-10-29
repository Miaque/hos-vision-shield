"""Exercise the authorized forwarded service API; never invokes SSH or changes live tasks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import av
import httpx

assert len(sys.argv) == 6, 'Usage: replay.py NAME normal|host a|b|short|a,b OUTPUT_ROOT RECORDINGS_ROOT'
root = Path(sys.argv[4]).resolve()
recordings = Path(sys.argv[5]).resolve()
name, mode, selection = sys.argv[1:4]
assert mode in ('normal', 'host')
clips = selection.split(',')
assert clips in (['a'], ['b'], ['short'], ['a', 'b'])
directory = root/'server-api'/name
directory.mkdir(parents=True, exist_ok=False)
inputs = dict(a=recordings/'rfm-live-20261027-150312/rfm-original-preview.mp4',
              b=recordings/'rfm-live-20261027-150915/rfm-original-preview.mp4',
              short=root/'short-b.mp4')
headers = {'X-Detection-Interval':'3', 'X-Tracking-Mode':'flow'}
if mode == 'host':
    headers['X-Detection-Profile'] = 'host'

def media(path):
    with av.open(str(path)) as reader:
        frames = [(f.width, f.height, str(f.pts*f.time_base)) for f in reader.decode(video=0)]
    with av.open(str(path)) as reader:
        audio = [(str(p.pts*p.time_base), str(p.dts*p.time_base), str(p.duration*p.time_base), hashlib.sha256(bytes(p)).hexdigest())
                 for p in reader.demux(audio=0) if p.dts is not None] if reader.streams.audio else []
    return frames, audio

with httpx.Client(base_url='http://localhost:18000', trust_env=False, timeout=180) as client:
    client.get('/health/ready').raise_for_status()
    def submit(clip):
        with inputs[clip].open('rb') as source:
            response = client.post('/v1/file-tasks/uploads', headers={**headers,'Idempotency-Key':f'plate-p0-{name}-{clip}'},
                                   data={'kind':'video','redaction_mode':'solid'}, files={'file':(clip+'.mp4',source,'video/mp4')})
        response.raise_for_status()
        return response.json()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(clips)) as pool:
        tasks = dict(zip(clips, pool.map(submit, clips)))
    (directory/'request.json').write_text(json.dumps(dict(name=name, mode=mode, headers=headers,
        inputs={c:dict(path=str(inputs[c]),sha256=hashlib.sha256(inputs[c].read_bytes()).hexdigest(),task_id=tasks[c]['id']) for c in clips},
        scope='Existing live tasks remain running. These mixed-load measurements are separate from exclusive-card acceptance.'),indent=2), 'utf-8')
    print('START '+json.dumps({c:t['id'] for c,t in tasks.items()}),flush=True)
    completed = {}
    with (directory/'observations.jsonl').open('x',encoding='utf-8') as log:
        next_progress = 0
        while len(completed)<len(tasks):
            assert time.monotonic()-started < 2400, 'Service replay exceeded bounded 40-minute wait'
            observations = {}
            for clip, task in tasks.items():
                if clip in completed:
                    continue
                response = client.get('/v1/tasks/'+task['id'])
                response.raise_for_status()
                current = response.json()
                observations[clip] = dict(task=current, headers={k:v for k,v in response.headers.items() if k in ('server-timing','x-detection-counts','x-video-decoder','x-video-encoder')})
                if current['state'] in ('succeeded','failed','cancelled'):
                    assert current['state']=='succeeded', current
                    assert current['stats']['processed_frames']=={'a':2208,'b':2230,'short':399}[clip]
                    assert current['stats']['dropped_frames']==0
                    completed[clip] = observations[clip]
            live = client.get('/v1/tasks')
            live.raise_for_status()
            live = {g:[{k:t[k] for k in ('id','kind','state','stats')} for t in items] for g,items in live.json().items()}
            for items in live.values():
                for task in items:
                    stream=task['stats'].get('stream')
                    if stream:
                        task['stats']['stream']={k:v for k,v in stream.items() if k!='recent_outputs'}
            resource = client.get('/v1/diagnostics/resources')
            resource.raise_for_status()
            log.write(json.dumps(dict(unix=time.time(),tasks=observations,live=live,resources=resource.json()))+'\n')
            log.flush()
            if time.monotonic()>=next_progress:
                print('PROGRESS '+json.dumps({c:t['task']['stats'] for c,t in observations.items()}),flush=True)
                next_progress=time.monotonic()+30
            if len(completed)<len(tasks):
                time.sleep(2)
    for clip, entry in completed.items():
        task=entry['task']
        if mode=='host':
            response=client.get('/v1/tasks/'+task['id']+'/diagnostics')
            response.raise_for_status()
            report=response.json()
            assert report['mode']=='host' and report['state']=='complete'
            (directory/(clip+'-diagnostics.json')).write_text(json.dumps(report,indent=2),'utf-8')
        target=directory/(clip+'-redacted.mp4')
        with client.stream('GET','/v1/tasks/'+task['id']+'/result') as response, target.open('xb') as output:
            response.raise_for_status()
            for data in response.iter_bytes():
                output.write(data)
        assert media(target)==media(inputs[clip]), 'Frame/PTS/AAC integrity changed'
        entry.update(fps=task['stats']['processed_frames']/task['stats']['elapsed_seconds'],
                     integrity='All dimensions/frame PTS and AAC payloads/clocks equal input',
                     output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print('COMPLETE '+json.dumps(dict(clip=clip,fps=entry['fps'],stats=task['stats'])),flush=True)
    (directory/'outcome.json').write_text(json.dumps(dict(mode=mode,clips=completed,request_wall_seconds=time.monotonic()-started),indent=2),'utf-8')
