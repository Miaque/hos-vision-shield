"""Use completed-job cumulative counters for RTSP waits; retained tasks are examples only."""
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'006-stream-backlog-pacing/experiments'))
from e0_analyze import difference, verify_runtime


def partition(after,before):
    totals=difference(after,before)
    wait=totals.get('wait_seconds',0)
    assert abs(wait-sum(totals.get(k,0) for k in (
        'wait_queue_dispatch_seconds','wait_worker_seconds','wait_after_worker_seconds')))<1e-6
    assert abs(totals.get('wait_worker_seconds',0)-totals.get('wait_unattributed_worker_seconds',0)-
               sum(v for k,v in totals.items() if k.startswith('wait_stage:')))<1e-6
    return totals


def analyze(rows,ref):
    states=defaultdict(list);ends={};tasks={}
    for row in rows:
        data=row.get('data',{})
        if row['kind']=='runtime': verify_runtime(data,ref)
        if row['kind']=='e0_state': states[row['attempt_id']].append(data)
        if row['kind']=='attempt' and data.get('e0'): ends[data['attempt_id']]=data['e0']
        if row['kind']=='primary_task':
            key=row['attempt_id'],data['task_id']
            if data['revision']>tasks.get(key,{}).get('revision',0): tasks[key]=data
    assert states,'No RTSP E0 snapshots'
    t0=min(g[0]['sampled_monotonic'] for g in states.values())
    t1=max(g[-1]['sampled_monotonic'] for g in states.values())
    result={}
    for aid,group in states.items():
        first,last=group[0],group[-1]
        if aid in ends and ends[aid]['ended_monotonic']<=t1: last=ends[aid]
        before={} if first['started_monotonic']>=t0 else first
        totals=partition(last['primary_task_totals'],before.get('primary_task_totals',{}))
        lo=before.get('sampled_monotonic',first['started_monotonic'])
        hi=last.get('ended_monotonic') or last['sampled_monotonic']
        selected=[r for (a,_),r in tasks.items() if a==aid and lo<r['result_exit']<=hi]
        assert len(selected)<=totals.get('completed',0),'Task coverage exceeds completed counters'
        groups=defaultdict(lambda:dict(retained_tasks=0,wait_seconds=0.))
        for row in selected:
            key=str((row.get('route'),row['kind'],row['full_frame'],row['input_shape'],(row.get('call') or {}).get('operation')))
            groups[key]['retained_tasks']+=1
            groups[key]['wait_seconds']+=row['partition']['wait_seconds']
        wait=totals.get('wait_seconds',0)
        result[aid]=dict(completed_job_window=dict(start=lo,end=hi),totals=totals,
            wait_fraction={k:v/wait for k,v in totals.items() if k.startswith('wait_') and k.endswith('_seconds')} if wait else {},
            retained_task_records=len(selected),missing_task_records=totals.get('completed',0)-len(selected),
            retained_only_groups=dict(groups),
            in_flight_at_start=before.get('primary_task_totals',{}).get('submitted',0)-before.get('primary_task_totals',{}).get('completed',0)-before.get('primary_task_totals',{}).get('submit_failed',0),
            in_flight_at_end=last['primary_task_totals'].get('submitted',0)-last['primary_task_totals'].get('completed',0)-last['primary_task_totals'].get('submit_failed',0))
    return dict(ref=ref,attempts=result,limits=[
        'Cumulative values cover completed quality primary jobs, even when records are evicted',
        'Endpoint differences include whole jobs completing in the window; not partial active main scopes',
        'Queue/dispatch includes wrapper and OS dispatch; after-worker includes observation, publication and waking',
        'Nested worker phases partition worker wait; concurrent worker totals and main overlap are not added',
        'Forward host wall includes GPU and implicit synchronization; per-thread CPU is not all host CPU',
        'Other executors are not correlated; caller source_pts/frame_sequence may trigger review of older lookahead pixels'])


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:
        totals=partition(dict(wait_seconds=3.,wait_queue_dispatch_seconds=.5,wait_worker_seconds=2.,
             wait_after_worker_seconds=.5,wait_unattributed_worker_seconds=.2,
             **{'wait_stage:prepare-primary_seconds':.3,'wait_stage:forward-primary-plate_seconds':1.5}),{})
        assert totals['wait_seconds']==3
        print('Completed-job partition self-check passed')
    else:
        path=Path(sys.argv[1]);result=analyze([json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()],sys.argv[2])
        path.with_name('primary-tasks-summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
