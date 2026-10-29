"""CPU fixture for incremental E0 primary-task observation; never creates a server task."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

from vision_shield.media.stream_e0 import StreamE0


def run(enabled, *, frames=3000, memory=False):
    if memory: tracemalloc.start()
    trace=StreamE0()
    started,cpu=time.perf_counter(),time.process_time()
    def model(value):
        with trace.span('stage:prepare-primary'): value+=1
        with trace.span('stage:forward-primary-plate'): value+=sum(range(40))
        with trace.span('stage:post-primary'): value+=1
        return value
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i in range(frames):
            future=trace.submit_primary(pool,model,i) if enabled else pool.submit(model,i)
            with trace.span('stage:prepare-detail'): sum(range(20))
            with trace.span('stage:forward-detail-plate'): sum(range(40))
            with trace.span('stage:wait-primary'):
                value=trace.primary_result(future) if enabled else future.result()
            assert value==i+782
            if i%100==0:
                assert len(json.dumps(trace.snapshot()).encode())<=128*1024
    result=dict(enabled=enabled,frames=frames,memory_run=memory,
                wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                completed=trace.primary_task_totals['completed'])
    if memory:
        result['retained_bytes'],result['peak_bytes']=tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result


if __name__=='__main__':
    directory=Path(sys.argv[1]); directory.mkdir(parents=True,exist_ok=False)
    results=[]
    for i in range(3):
        for enabled in ([False,True] if i%2==0 else [True,False]):
            results.append(run(enabled))
    for enabled in (False,True): results.append(run(enabled,memory=True))
    medians={str(enabled):{key:statistics.median(r[key] for r in results if
                 r['enabled']==enabled and not r['memory_run']) for key in ('wall_seconds','cpu_seconds')}
             for enabled in (False,True)}
    report=dict(runs=results,medians=medians,
                limitations='CPU fixture, same E0 scopes with primary hooks off/on; no GPU, Redis or HTTP cost claim')
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
