"""Read-only competing-stream/resource sidecar; never creates, stops or changes a task."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import httpx


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--ref',default='41ba7e3210cc9d8e0d2a34c06d77be49ef2ee0f5')
    args=parser.parse_args()
    assert 1<=args.seconds<=1800
    args.directory.mkdir(parents=True,exist_ok=False)
    ref=args.ref
    task_id='2d04c6fd-45ce-4ffa-8c6e-b0ae1b9cba4f'
    started,last_extra,last_print=time.monotonic(),0,0
    seen,seq,attempts,expected_cfg=set(),{}, {},None
    failed,written,last_rtp=0,0,started
    with httpx.Client(base_url='http://127.0.0.1:18000',trust_env=False,timeout=8) as client, \
            (args.directory/'e0.jsonl').open('x',encoding='utf-8') as output:
        def save(kind,**data):
            global written
            line=json.dumps(dict(kind=kind,captured_unix=time.time(),**data),separators=(',',':'))+'\n'
            written+=len(line.encode())
            if written>64*1024**2:
                raise RuntimeError('Sidecar byte limit')
            output.write(line)
            output.flush()
        while time.monotonic()-started<args.seconds:
            tick=time.monotonic()
            try:
                response=client.get('/v1/tasks/'+task_id+'/diagnostics')
                response.raise_for_status()
                data=response.json()
                live=data['live']; aid=live['attempt_id']; runtime=live['runtime']
                if aid not in seen:
                    assert len(runtime['source_sha256'])==20
                    for name,sha in runtime['source_sha256'].items():
                        blob=subprocess.check_output(['git','show',ref+':src/vision_shield/'+name.replace('.','/')+'.py'])
                        assert hashlib.sha256(blob).hexdigest()==sha,'Peer runtime mismatch'
                    if expected_cfg is None:
                        expected_cfg=runtime['effective_config']
                    assert expected_cfg==runtime['effective_config'],'Peer configuration changed'
                    save('runtime',attempt_id=aid,data=runtime)
                    seen.add(aid)
                state=live['e0']
                for key in ('frames','pacer','audio_clocks','video_clocks','audio_outputs','events'):
                    state.pop(key,None)
                save('e0_state',attempt_id=aid,data=state)
                for attempt in data.get('attempts',[]):
                    stamp=hashlib.sha256(json.dumps(attempt,sort_keys=True).encode()).hexdigest()
                    if attempts.get(attempt['run_id'])!=stamp:
                        save('attempt',data=attempt)
                        attempts[attempt['run_id']]=stamp
                receiver=data.get('receiver')
                if receiver:
                    assert receiver['clock']==runtime['clock_identity']
                    key=receiver['observer_id']
                    for frame in receiver.pop('frames'):
                        if frame['sequence']>seq.get(key,0):
                            save('rtp_frame',observer_id=key,data=frame)
                            seq[key]=frame['sequence'];last_rtp=time.monotonic()
                    save('receiver_state',data=receiver)
                assert time.monotonic()-last_rtp<15,'Peer RTP absent for 15s'
                if tick-last_extra>=5:
                    response=client.get('/v1/tasks');response.raise_for_status()
                    active=response.json()['current']+response.json()['queued']
                    save('active_tasks',data=[{k:t[k] for k in ('id','kind','state','ready','desired_state')} for t in active])
                    assert all(t['kind']=='stream' for t in active),'Additional file load invalidates clean baseline'
                    response=client.get('/v1/diagnostics/resources');response.raise_for_status()
                    save('resources',data=response.json())
                    last_extra=tick
                failed=0
            except httpx.HTTPError as error:
                failed+=1
                save('http_error',error_type=type(error).__name__)
                if failed>=3:
                    raise RuntimeError('Three sidecar HTTP failures') from None
            if tick-last_print>=30:
                print(json.dumps(dict(seconds=round(tick-started,1),bytes=written)),flush=True)
                last_print=tick
            time.sleep(max(0,2-(time.monotonic()-tick)))
    print('Read-only peer/resource capture complete',flush=True)
