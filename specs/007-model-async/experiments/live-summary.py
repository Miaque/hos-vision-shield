"""Summarize existing dual RTSP observations; file-task throughput is excluded."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'006-stream-backlog-pacing/experiments'))
from e0_analyze import budget,transported_fingerprint
from analyze import distribution,low_fps


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def stream(rows):
    states=defaultdict(list); ends={}; identities={}
    for row in rows:
        if row['kind']=='runtime':
            identities[transported_fingerprint(row['data'])]=row['data']
        if row['kind']=='e0_state':
            states[row['attempt_id']].append(row['data'])
        if row['kind']=='attempt' and row['data'].get('e0'):
            ends[row['data']['attempt_id']]=row['data']['e0']
    t0=min(g[0]['sampled_monotonic'] for g in states.values())
    t1=max(g[-1]['sampled_monotonic'] for g in states.values())
    result={}
    for aid,group in states.items():
        first,last=group[0],group[-1]
        if aid in ends and ends[aid]['ended_monotonic']<=t1:
            last=ends[aid]
        assert first['runtime_fingerprint'] in identities and last['runtime_fingerprint'] in identities
        previous=None if first['started_monotonic']>=t0 else first
        b=budget(last,previous); n=b['worker_mux_packets']; w=b['wall_seconds']; main=b['main_exclusive_seconds']
        processing=sum(v for k,v in main.items() if k.startswith('stage:') or k in (
            'op:process','op:render','op:lookahead-push','op:lookahead-expire','op:lookahead-flush','op:lookahead-discover'))
        worker=Counter()
        for tid,values in last.get('other_thread_exclusive_seconds',{}).items():
            before=(previous or {}).get('other_thread_exclusive_seconds',{}).get(tid,{})
            for key,value in values.items():
                worker[key]+=value-before.get(key,0)
        result[aid]=dict(wall_seconds=w,mux_packets=n,mux_fps=n/w,media_progress=b['media_progress'],
            pacer=b['pacer'],clock=b['clock'],fifo=b['fifo'],
            nonpacer_wall_ms_per_frame=b['non_pacer_wall_ms_per_mux_packet'],
            processing_lookahead_sync_ms_per_frame=processing*1000/n if n else None,
            main_exclusive_ms_per_frame={k:v*1000/n for k,v in main.items()} if n else {},
            other_threads_completed_exclusive_ms_per_frame={k:v*1000/n for k,v in worker.items()} if n else {},
            fifo_full_fraction=b['fifo']['full_occupancy_including_active_seconds']/w,
            fifo_mean_wait_ms=1000*b['fifo'].get('queue_wait_seconds',0)/b['fifo']['dequeued'] if b['fifo'].get('dequeued') else None,
            rebase_only_arithmetic_fps=n/(w-b['pacer'].get('conditional_rebase_wait_seconds',0)),
            note='Worker scopes overlap the main thread and cannot be added to its budget; arithmetic FPS is not an E1 result')
    rtp=[r for r in rows if r['kind']=='rtp_frame' and t0<=r['data']['received_monotonic']<=t1]
    times=[r['data']['received_monotonic'] for r in rtp]
    gaps=[b['data']['received_monotonic']-a['data']['received_monotonic'] for a,b in zip(rtp,rtp[1:])
          if (a['observer_id'],a['data']['connection'])==(b['observer_id'],b['data']['connection'])
          and b['data']['sequence']==a['data']['sequence']+1]
    connection_gaps=[dict(seconds=b['data']['received_monotonic']-a['data']['received_monotonic'],
                         after_monotonic=a['data']['received_monotonic'],before_monotonic=b['data']['received_monotonic'])
        for a,b in zip(rtp,rtp[1:]) if (a['observer_id'],a['data']['connection'])!=(b['observer_id'],b['data']['connection'])]
    steps=Counter((b['data']['rtp_timestamp']-a['data']['rtp_timestamp'])%(2**32) for a,b in zip(rtp,rtp[1:])
                  if (a['observer_id'],a['data']['connection'])==(b['observer_id'],b['data']['connection']))
    total=Counter()
    for a in result.values():
        total.update(a['pacer'])
    return dict(server_window=dict(start=t0,end=t1,seconds=t1-t0),attempts=result,pacer_totals=dict(total),
        rtp=dict(frames=len(times),effective_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            interval_seconds=distribution(gaps),over_500ms=sum(g>.5 for g in gaps),over_1000ms=sum(g>1 for g in gaps),
            low_1s_fps=low_fps(times,1),low_5s_fps=low_fps(times,5),connection_gaps=connection_gaps,
            most_common_rtp_timestamp_steps=steps.most_common(5),
            receipt_sequence_gaps=sum(a['observer_id']==b['observer_id'] and b['data']['sequence']!=a['data']['sequence']+1
                                      for a,b in zip(rtp,rtp[1:]))),
        runtime=list(identities.values()),client_start_unix=min(r['captured_unix'] for r in rows),
        client_end_unix=max(r['captured_unix'] for r in rows))


if __name__=='__main__':
    root=Path(sys.argv[1]); primary=read(root/'primary/e0.jsonl'); peer=read(root/'peer/e0.jsonl')
    resources=[r['data'] for r in peer if r['kind']=='resources']
    gpu={};cpu=defaultdict(list)
    for sample in resources:
        latest={}
        for worker in sample['workers']:
            if worker['stale'] or not worker.get('snapshot'):
                continue
            snapshot=worker['snapshot']; key=(worker['pool'],worker['worker_id'])
            if snapshot['cpu']['percent'] is not None:
                cpu[key].append(snapshot['cpu']['percent'])
            for device in snapshot['gpu']['devices']:
                old=latest.get(device['uuid'])
                if old is None or old[0]<snapshot['sampled_at_unix']:
                    latest[device['uuid']]=(snapshot['sampled_at_unix'],device)
        for uuid,(_,device) in latest.items():
            gpu[(sample['sampled_at_unix'],uuid)]=device
    values=list(gpu.values())
    tasks=[r['data'] for r in peer if r['kind']=='active_tasks']
    output=dict(primary=stream(primary),peer=stream(peer),
        active_task_samples=len(tasks),observed_active_ids=sorted({t['id'] for group in tasks for t in group}),
        file_task_samples=sum(any(t['kind']!='stream' for t in group) for group in tasks),
        resources=dict(device_utilization_percent=distribution([d['utilization_percent'] for d in values if d['utilization_percent'] is not None]),
            device_memory_bytes=distribution([d['memory_used_bytes'] for d in values if d['memory_used_bytes'] is not None]),
            worker_cpu_percent={str(k):distribution(v) for k,v in cpu.items()},
            note='GPU is device-wide shared utilization, not per-stream attribution; task-to-worker ownership is not exposed by these APIs'),
        limits=['RTP receiver records H264 pictures, not audio RTP/RTCP or player rendering',
                'Clean live capture starts after both experimental file tasks completed; file intervals are excluded',
                'Only HTTP evidence; no SSH probes, source changes, task replacement or further deployment'])
    (root/'live-summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(primary_fps=output['primary']['rtp']['effective_fps'],peer_fps=output['peer']['rtp']['effective_fps'],
        gpu=output['resources']['device_utilization_percent'],file_samples=output['file_task_samples'])))
