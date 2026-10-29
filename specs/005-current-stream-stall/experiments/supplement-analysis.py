"""Summarize only attempts with a verified runtime, preserving all time bases."""
from collections import Counter, defaultdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analysis', ROOT / 'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def numeric(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def q(f):
        position = (len(values)-1)*f
        left = int(position)
        return values[left] + (values[min(left+1, len(values)-1)]-values[left])*(position-left)
    return dict(n=len(values), min=values[0], p50=q(.5), p95=q(.95), p99=q(.99), max=values[-1], mean=sum(values)/len(values))


def build(name):
    folder = ROOT / name
    phase = json.loads((folder/'phase-analysis.json').read_text())
    rows = analysis.read(folder/'live.jsonl')
    valid = set(phase['runtime'])
    tasks = [x for x in rows if x['kind']=='task' and x['data'].get('stats',{}).get('stream',{}).get('attempt_id') in valid]
    frames = list({(x['attempt_id'],x['sequence']):x for x in rows if x['kind']=='frame' and x['attempt_id'] in valid}.values())
    operations = list({(x['attempt_id'],x['sequence']):x for x in rows if x['kind']=='operation' and x['attempt_id'] in valid}.values())
    live = [x for x in rows if x['kind']=='live_state' and x['attempt_id'] in valid]
    queues = {key:numeric([x['data']['stats']['stream'][key] for x in tasks if key in x['data']['stats']['stream']])
              for key in ('queue_frames','lookahead_frames','audio_queue_bytes','pending_encoder_frames')}
    ages = {key:numeric([x[key] for x in live if x.get(key) is not None])
            for key in ('oldest_input_age_seconds','oldest_lookahead_wait_seconds')}
    counter_keys = ('input_frames','encoded_frames','unmatched_packets','missing_receive_frames','failed_frames','timestamp_repairs')
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['data']['stats']['stream']['attempt_id']].append(task)
    counters, timing = {}, {}
    def timings(row):
        return {name.removeprefix('task-'):float(value)/1000 for item in row.get('headers',{}).get('server-timing','').split(', ')
                if ';dur=' in item for name,value in [item.split(';dur=')]}
    for aid, group in grouped.items():
        first, last = group[0]['data'],group[-1]['data']
        a,b = first['stats']['stream'],last['stats']['stream']
        counters[aid] = {key:dict(first=a.get(key),last=b.get(key)) for key in counter_keys}
        counters[aid]['dropped_frames'] = dict(first=first['stats'].get('dropped_frames'),last=last['stats'].get('dropped_frames'))
        elapsed = b['sampled_monotonic']-a['sampled_monotonic']
        before,after = timings(group[0]),timings(group[-1])
        timing[aid] = dict(window_seconds=elapsed, output_frames=b['encoded_frames']-a['encoded_frames'],
                          delta_seconds={key:value-before.get(key,0) for key,value in after.items()})
    resources = [x for x in rows if x['kind']=='resources' and tasks[0]['captured_unix']<=x['captured_unix']<=tasks[-1]['captured_unix']]
    workers,devices = {},{}
    for sample in resources:
        for worker in sample['data'].get('workers',[]):
            if worker.get('stale') or not worker.get('snapshot'):
                continue
            snap = worker['snapshot']
            if worker['pool']=='streams':
                workers[(worker['worker_id'],snap.get('sampled_at_unix'))] = snap
            for device in snap.get('gpu',{}).get('devices',[]):
                devices[(device['uuid'],snap.get('sampled_at_unix'))] = device
    resource_stats = dict(stream_worker_cpu_percent=numeric([x['cpu']['percent'] for x in workers.values() if x['cpu'].get('percent') is not None]),
                          stream_worker_memory_bytes=numeric([x['memory']['used_bytes'] for x in workers.values()]),
                          gpu_utilization_percent=numeric([x['utilization_percent'] for x in devices.values()]),
                          gpu_memory_bytes=numeric([x['memory_used_bytes'] for x in devices.values()]))
    repairs = [x for x in frames if x['timestamp_repaired']]
    raw_pts_diff = []
    for a,b in zip(frames,frames[1:]):
        if a['attempt_id']==b['attempt_id'] and b['sequence']==a['sequence']+1:
            raw_pts_diff.append(float(Fraction(b['source_pts'])-Fraction(a['source_pts'])))
    result = dict(queues=queues,ages=ages,counters=counters,timing=timing,resources=resource_stats,
                  sample_count=len(tasks),queue_full_fraction=sum(x['data']['stats']['stream'].get('queue_frames')==2 for x in tasks)/len(tasks),
                  raw_source_pts_deltas=numeric(raw_pts_diff),source_pts_nonpositive=sum(x<=0 for x in raw_pts_diff),
                  timestamp_repaired_frames=[{k:x[k] for k in ('attempt_id','sequence','source_pts','timeline_pts')} for x in repairs],
                  stage_operation_totals={key:sum(x['seconds'] for x in operations if x['operation_kind']==key) for key in sorted({x['operation_kind'] for x in operations})},
                  note='only process/render operations are exhaustive; other operation totals contain samples >=100ms only')
    (folder/'supplement-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,json.dumps({k:result[k] for k in ('queues','ages','queue_full_fraction','source_pts_nonpositive','resources')},ensure_ascii=False))


if __name__=='__main__':
    assert numeric([1,2,3])['p50']==2
    build(sys.argv[1])
