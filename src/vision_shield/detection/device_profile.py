"""Opt-in, three-call CUDA tracing; never use sampled task FPS as an optimization result."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter


CHECKPOINTS = (64, 424, 768)
SCOPE = "vision-shield-primary-plate"
MAX_TRACE_BYTES = 32 * 1024**2


def busy_ms(events):
    end, total = float('-inf'), 0.0
    for event in sorted(events, key=lambda item: item['ts']):
        start, stop = event['ts'], event['ts'] + event['dur']
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1000


def summarize(trace):
    events = [event for event in trace['traceEvents'] if event.get('ph') == 'X']
    scopes = [event for event in events if event.get('cat') == 'user_annotation' and event['name'] == SCOPE]
    if len(scopes) != 1:
        raise RuntimeError('Missing primary plate trace scope')
    scope = scopes[0]
    runtime = [event for event in events if event.get('cat') in {'cuda_runtime', 'cuda_driver'}
               and event['tid'] == scope['tid'] and scope['ts'] <= event['ts'] < scope['ts'] + scope['dur']]
    correlations = {event['args']['correlation'] for event in runtime if 'correlation' in event.get('args', {})}
    gpu = [event for event in events if event.get('cat') in {'kernel', 'gpu_memcpy', 'gpu_memset'}
           and event.get('args', {}).get('correlation') in correlations]
    kernels = [event for event in gpu if event['cat'] == 'kernel']
    if not kernels:
        raise RuntimeError('CUDA trace contains no correlated primary kernels')
    groups = defaultdict(lambda: dict(calls=0, milliseconds=0.0))
    for event in kernels:
        group = groups[event['name']]
        group['calls'] += 1
        group['milliseconds'] += event['dur'] / 1000
    result = dict(host_forward_ms=scope['dur'] / 1000, primary_thread_id=scope['tid'],
                  kernel_count=len(kernels), kernel_sum_ms=sum(event['dur'] for event in kernels) / 1000,
                  gpu_busy_ms=busy_ms(gpu), cuda_api_events=len(runtime),
                  synchronization_api_events=sum('synchronize' in event['name'].lower() for event in runtime),
                  synchronization_host_ms=busy_ms([event for event in runtime if 'synchronize' in event['name'].lower()]),
                  allocation_host_ms=busy_ms([event for event in runtime if any(word in event['name'].lower() for word in ('malloc', 'free'))]),
                  top_kernels=[dict(name=name[:256], name_truncated=len(name) > 256, **values) for name, values in
                               sorted(groups.items(), key=lambda item: item[1]['milliseconds'], reverse=True)[:12]])
    for direction in ('HtoD', 'DtoH', 'DtoD'):
        copies = [event for event in gpu if event['cat'] == 'gpu_memcpy' and direction.lower() in event['name'].lower()]
        result[direction] = dict(calls=len(copies), milliseconds=sum(event['dur'] for event in copies) / 1000,
                                 bytes=sum(event.get('args', {}).get('bytes', 0) for event in copies))
    return result


def cpu_operators(events):
    scope, = [event for event in events if event.name == SCOPE and event.device_type.name == 'CPU']
    groups = defaultdict(lambda: dict(calls=0, self_cpu_ms=0.0, self_device_ms=0.0))
    for event in events:
        if (event.device_type.name != 'CPU' or event.thread != scope.thread or event.name == SCOPE
                or event.time_range.start < scope.time_range.start or event.time_range.end > scope.time_range.end):
            continue
        group = groups[event.name]
        group['calls'] += 1
        group['self_cpu_ms'] += event.self_cpu_time_total / 1000
        group['self_device_ms'] += event.self_device_time_total / 1000
    return [dict(name=name[:256], name_truncated=len(name) > 256, **values) for name, values in
            sorted(groups.items(), key=lambda item: item[1]['self_cpu_ms'], reverse=True)[:12]]


class DeviceSamples:
    def __init__(self, directory, emit):
        self.directory, self.emit = Path(directory), emit
        self.frames, self.samples = 0, []
        self.previous_forward_ms = None

    def forward(self, model, pixels, *, global_view):
        if not global_view:
            return model(pixels)
        self.frames += 1
        if self.frames not in CHECKPOINTS:
            started = perf_counter()
            result = model(pixels)
            self.previous_forward_ms = (perf_counter() - started) * 1000
            return result
        import torch
        from torch.profiler import ProfilerActivity, profile, record_function

        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                with record_function(SCOPE):
                    result = model(pixels)
            target = self.directory / f'plate-profile-{self.frames}.json'
            temporary = target.with_suffix('.json.part')
            profiler.export_chrome_trace(str(temporary))
            if temporary.stat().st_size > MAX_TRACE_BYTES:
                temporary.unlink()
                raise RuntimeError('Device trace exceeded budget')
            temporary.replace(target)
            summary = summarize(json.loads(target.read_text('utf-8')))
            summary.update(frame=self.frames, trace_bytes=target.stat().st_size,
                           preceding_unprofiled_forward_ms=self.previous_forward_ms,
                           gpu_name=torch.cuda.get_device_name(pixels.device), input_dtype=str(pixels.dtype),
                           input_shape=list(pixels.shape), input_device=str(pixels.device),
                           parameter_dtypes=sorted({str(parameter.dtype) for parameter in model.parameters()}),
                           top_cpu_operators=cpu_operators(profiler.events()))
            self.samples.append(summary)
            self.emit({'device_profile': dict(state='collecting', samples=self.samples)})
            return result
        except Exception:
            self.emit({'device_profile': dict(state='failed', reason='sampling_failed', samples=self.samples)})
            raise

    def finish(self):
        self.emit({'device_profile': dict(state='complete' if len(self.samples) == len(CHECKPOINTS) else 'partial',
                                         samples=self.samples)})
