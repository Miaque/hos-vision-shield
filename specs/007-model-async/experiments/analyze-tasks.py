"""Partition complete async-host observations; never add concurrent work to main wall."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def exclusive_segments(spans):
    """Subtract nested same-thread scopes while retaining each interval's location."""
    nodes, stack = [], []
    for row in sorted(spans, key=lambda r: (r['started'], -r['monotonic'], -r['sequence'])):
        if row['monotonic'] <= row['started']:
            continue
        node = [row, []]
        while stack and row['started'] >= stack[-1][0]['monotonic']:
            stack.pop()
        if stack:
            assert row['monotonic'] <= stack[-1][0]['monotonic'] + 1e-8, 'Crossed same-thread scopes'
            stack[-1][1].append(node)
        nodes.append(node)
        stack.append(node)
    for row, children in nodes:
        cursor = row['started']
        for child, _ in children:
            if child['started'] > cursor:
                yield cursor, child['started'], row
            cursor = child['monotonic']
        if cursor < row['monotonic']:
            yield cursor, row['monotonic'], row


def intersection(a, b, c, d):
    return max(0., min(b, d)-max(a, c))


def analyze(path, frames):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [r['sequence'] for r in rows] == list(range(1, len(rows)+1)), 'Incomplete observations'
    threads, jobs, calls = defaultdict(list), defaultdict(dict), {}
    for row in rows:
        if row['event']=='span':
            threads[row['thread_id']].append(row)
        if row['event'] in ('submit', 'worker_start', 'worker_return', 'future_done_callback'):
            jobs[row['job_id']][row['event']] = row
        if row['event']=='model_call':
            calls[row['model_call_id']] = row
    segments = [segment for spans in threads.values() for segment in exclusive_segments(spans)]
    main = next(r['thread_id'] for r in rows if r['event']=='span' and r['stage']=='frame-regions')
    main_segments = sorted((a,b,r) for a,b,r in segments if r['thread_id']==main)
    main_ends = [b for a,b,r in main_segments]
    job_segments = defaultdict(list)
    for a,b,r in segments:
        if r.get('job_id') is not None:
            job_segments[r['job_id']].append((a,b,r))
    exclusive = Counter()
    for a,b,r in segments:
        if r['thread_id']==main:
            exclusive[r['stage']] += b-a
    waits, overlap, queue, worker = Counter(), Counter(), 0., 0.
    primary_jobs = {jid:j for jid,j in jobs.items() if j['submit'].get('model')=='primary'}
    for jid,job in primary_jobs.items():
        assert all(k in job for k in ('worker_start','worker_return','future_done_callback')), 'Unfinished task'
        start, end = job['worker_start']['monotonic'], job['worker_return']['monotonic']
        queue += start-job['submit']['monotonic']
        worker += end-start
        for a,b,r in main_segments[bisect_right(main_ends,start):]:
            if a >= end:
                break
            if r['stage']!='wait-primary':
                overlap[r['stage']] += intersection(a,b,start,end)
    waiting = [r for r in rows if r['event']=='result_exit' and r['thread_id']==main
               and r['waited_job_id'] in primary_jobs]
    wait_groups = defaultdict(lambda: dict(calls=0,seconds=0.))
    for row in waiting:
        start,end,jid = row['started'],row['monotonic'],row['waited_job_id']
        submitted = primary_jobs[jid]['submit']
        group = wait_groups[str((submitted.get('kind'),submitted.get('full_frame'),submitted.get('purpose')))]
        group['calls'] += 1
        group['seconds'] += end-start
        waits['queue_dispatch'] += row['queue_dispatch_seconds']
        waits['publication_wakeup'] += row['after_worker_seconds']
        inside = 0.
        for a,b,r in job_segments[jid]:
            part = intersection(a,b,start,end)
            waits[r['stage']] += part
            inside += part
        waits['worker_uninstrumented'] += max(0., row['worker_seconds']-inside)
    wait_total = sum(r['monotonic']-r['started'] for r in waiting)
    assert abs(sum(waits.values())-wait_total) < 1e-5
    distribution = Counter((r['model_group'],r.get('candidate_kind'),r.get('purpose'),
                            tuple(r['source_shape']),r.get('mirror')) for r in calls.values())
    plate = defaultdict(lambda: dict(calls=0, host_seconds=0., thread_cpu_seconds=0.))
    for row in rows:
        if row['event']=='span' and row['stage'] in ('forward-primary-plate','forward-detail-plate'):
            call = calls[row['model_call_id']]
            key = (row['stage'],call.get('purpose'),tuple(call['source_shape']),call.get('mirror'))
            group = plate[str(key)]
            group['calls'] += 1
            group['host_seconds'] += row['wall_seconds']
            group['thread_cpu_seconds'] += row['thread_cpu_seconds']
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=frames, events=len(rows),
        primary_jobs=len(primary_jobs), primary_task_queue_dispatch_seconds=queue,
        primary_task_worker_seconds=worker, main_primary_wait_calls=len(waiting),
        main_primary_wait_seconds=wait_total, main_primary_wait_partition_seconds=dict(waits),
        main_primary_wait_groups=wait_groups,
        primary_overlap_with_main_seconds=dict(overlap),
        main_exclusive_ms_per_frame={k:v*1000/frames for k,v in exclusive.items()},
        model_call_distribution={str(k):v for k,v in distribution.items()}, plate_forward=plate,
        limits=['Host forward includes device work and implicit synchronization; no pure GPU attribution',
                'Model worker time overlaps main stages and must not be added to main wall',
                'Queue includes observer submission overhead and OS dispatch',
                'Complete-file timing is not live backlog or player verification'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('frames', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.trace, args.frames), indent=2), encoding='utf-8')
