"""CPU-only incremental observer cost; excludes GPU, Redis, HTTP and the E0 stream path."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import tracemalloc

from vision_shield.detection.async_profile import instrument
from vision_shield.detection.host_profile import ContextPool, ProfileTimes, scope


def run(directory, enabled, frames=1000, *, memory=False):
    if memory:
        tracemalloc.start()
    collector = ProfileTimes(directory, async_tasks=enabled)
    started, cpu = time.perf_counter(), time.process_time()
    with ContextPool(1) as pool, ExitStack() as stack:
        if enabled:
            instrument(stack, collector.async_tasks)
        for frame in range(frames):
            with scope(frame=frame), collector.measure('frame'):
                def work():
                    with collector.measure('forward-primary-plate'):
                        return sum(range(20))
                pending = pool.submit(work)
                for _ in range(24):
                    with collector.measure('prepare-detail'):
                        pass
                with collector.measure('wait-primary'):
                    assert pending.result()==190
    wall, active_cpu = time.perf_counter()-started, time.process_time()-cpu
    retained, peak = tracemalloc.get_traced_memory() if memory else (None,None)
    if memory:
        tracemalloc.stop()
    collector.close()
    return dict(enabled=enabled, frames=frames, measurement='memory' if memory else 'timing', wall_seconds=wall, cpu_seconds=active_cpu,
                retained_bytes=retained, peak_bytes=peak,
                async_bytes=collector.async_tasks.bytes if enabled else 0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    results = []
    for i in range(3):
        for enabled in ([False,True] if i%2==0 else [True,False]):
            results.append(run(args.directory/f'{i}-{enabled}', enabled))
    for enabled in (False,True):
        results.append(run(args.directory/f'memory-{enabled}', enabled, memory=True))
    (args.directory/'report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results))
