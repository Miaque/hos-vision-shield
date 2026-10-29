"""Opt-in file diagnostics for cached model pools; no device synchronization."""
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar
import hashlib
import json
from pathlib import Path
from threading import Lock, get_ident, local
import time
from unittest.mock import patch


JOB = ContextVar('vision_shield_async_job', default=None)


class AsyncProfile:
    def __init__(self, directory, *, byte_limit=64 * 1024**2):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/'async-tasks.jsonl'
        self.file = self.path.open('x', encoding='utf-8')
        self.lock, self.context = Lock(), local()
        self.limit, self.bytes, self.sequence, self.identity = byte_limit, 0, 0, 0
        self.main_thread, self.closed = get_ident(), False
        self.jobs, self.models = {}, {}
        self.counts, self.exclusive, self.cpu = Counter(), defaultdict(Counter), defaultdict(Counter)
        self.source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def new_id(self):
        with self.lock:
            self.identity += 1
            return self.identity

    def event(self, name, *, at=None, **data):
        from vision_shield.detection.host_profile import CONTEXT
        metadata = {k: v for k, v in CONTEXT.get().items() if k not in ('local_calls', 'local_details')}
        if name == 'span':
            metadata = {k:v for k,v in metadata.items() if k in ('frame', 'model_call_id')}
        row = dict(metadata, event=name, thread_id=get_ident(), job_id=JOB.get(),
                   monotonic=time.perf_counter() if at is None else at)
        row.update(data)
        with self.lock:
            if self.closed:
                self.counts['late_events_after_close'] += 1
                return
            self.sequence += 1
            line = json.dumps(dict(row, sequence=self.sequence), separators=(',', ':'))+'\n'
            self.bytes += len(line.encode())
            if self.bytes > self.limit:
                raise RuntimeError('Async diagnostic byte budget exceeded')
            self.file.write(line)

    @contextmanager
    def span(self, name):
        stack = getattr(self.context, 'stack', None)
        if stack is None:
            stack = self.context.stack = []
        item = [time.perf_counter(), time.thread_time(), 0., 0.]
        stack.append(item)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            end, cpu = time.perf_counter(), time.thread_time()
            wall, active_cpu = end-item[0], cpu-item[1]
            stack.pop()
            if stack:
                stack[-1][2] += wall
                stack[-1][3] += active_cpu
            self.event('span', at=end, stage=name, started=item[0], wall_seconds=wall,
                       thread_cpu_seconds=active_cpu, exclusive_seconds=max(0., wall-item[2]), failed=failed)
            with self.lock:
                self.exclusive[get_ident()][name] += max(0., wall-item[2])
                self.cpu[get_ident()][name] += max(0., active_cpu-item[3])

    def report(self):
        with self.lock:
            # Future callbacks log exceptions instead of propagating them to result().
            # A budget failure there must still prevent a successful partial report.
            if self.bytes > self.limit:
                raise RuntimeError('Async diagnostic byte budget exceeded')
            if not self.closed:
                self.file.flush()
            return dict(jobs=len(self.jobs), **dict(self.counts), bytes=self.bytes,
                source_sha256=self.source_hash,
                main_exclusive_seconds=dict(self.exclusive[self.main_thread]),
                main_exclusive_thread_cpu_seconds=dict(self.cpu[self.main_thread]),
                worker_exclusive_seconds={str(t): dict(v) for t,v in self.exclusive.items() if t != self.main_thread},
                phase_boundaries='host wall and per-thread CPU; forward wall includes GPU/implicit synchronization',
                limits=dict(bytes=self.limit, jobs=25000),
                clock=dict(name='perf_counter', **vars(time.get_clock_info('perf_counter'))),
                notes=['Queue/dispatch includes executor enqueue, thread startup and OS dispatch; not pure queue residence',
                       'worker_return precedes Future state publication; done callback may run after result returns',
                       'Wait intersections partition one main wait; do not add concurrent model durations',
                       'Per-thread CPU is active host CPU; wall minus CPU is not pure GPU time'])

    def close(self):
        with self.lock:
            self.closed = True
            self.file.close()


def instrument(stack, trace):
    """Install only in the isolated explicit diagnostic process, including existing pools."""
    from vision_shield.detection.host_profile import CONTEXT, scope
    original_submit, original_result = ThreadPoolExecutor.submit, Future.result

    def submit(pool, fn, /, *args, **kwargs):
        jid, requested = trace.new_id(), time.perf_counter()
        contextual = (isinstance(getattr(fn, '__self__', None), Context)
                      and getattr(fn, '__name__', '') == 'run' and args and callable(args[0]))
        target, inputs = (args[0], args[1:]) if contextual else (fn, args)
        owner = getattr(target, '__self__', None)
        model = owner if owner is not None else inputs[0] if inputs else None
        label = getattr(target, '__name__', type(target).__name__)
        metadata = dict(CONTEXT.get())
        job = dict(submit=requested, function=label, model=trace.models.get(id(model)),
                   kind=kwargs.get('kind'), full_frame=kwargs.get('full_frame'), parent_job=JOB.get())
        with trace.lock:
            if len(trace.jobs) >= 25000:
                raise RuntimeError('Async diagnostic job budget exceeded')
            trace.jobs[jid] = job
        trace.event('submit', at=requested, job_id=jid, **job)

        def execute():
            token = JOB.set(jid)
            with scope(**metadata):
                start, cpu = time.perf_counter(), time.thread_time()
                job['worker_start'] = start
                trace.event('worker_start', at=start)
                try:
                    return target(*inputs, **kwargs)
                except BaseException:
                    job['failed'] = True
                    raise
                finally:
                    end = time.perf_counter()
                    job['worker_return'] = end
                    trace.event('worker_return', at=end, thread_cpu_seconds=time.thread_time()-cpu,
                                failed=job.get('failed', False))
                    with trace.lock:
                        trace.counts['queue_dispatch_seconds'] += start-requested
                        trace.counts['worker_seconds'] += end-start
                        trace.counts['failed_jobs'] += job.get('failed', False)
                    JOB.reset(token)

        def run():
            return fn(execute) if contextual else execute()

        future = original_submit(pool, run)
        future._vision_shield_profile = (trace, jid)
        def done(value):
            trace.event('future_done_callback', job_id=jid, cancelled=value.cancelled())
        future.add_done_callback(done)
        return future

    def result(future, *args, **kwargs):
        identity = getattr(future, '_vision_shield_profile', None)
        if identity is None or identity[0] is not trace:
            return original_result(future, *args, **kwargs)
        jid, start = identity[1], time.perf_counter()
        trace.event('result_enter', at=start, waited_job_id=jid)
        try:
            return original_result(future, *args, **kwargs)
        finally:
            end = time.perf_counter()
            job = trace.jobs[jid]
            worker_start = job.get('worker_start', end)
            worker_end = job.get('worker_return', end)
            queue = max(0., min(end, worker_start)-start)
            execution = max(0., min(end, worker_end)-max(start, worker_start))
            after = max(0., end-start-queue-execution)
            trace.event('result_exit', at=end, waited_job_id=jid, started=start,
                        queue_dispatch_seconds=queue, worker_seconds=execution, after_worker_seconds=after)
            with trace.lock:
                trace.counts['wait_calls'] += 1
                trace.counts['wait_seconds'] += end-start
                trace.counts['queue_dispatch_inside_wait_seconds'] += queue
                trace.counts['worker_inside_wait_seconds'] += execution
                trace.counts['after_worker_inside_wait_seconds'] += after

    stack.enter_context(patch.object(ThreadPoolExecutor, 'submit', submit))
    stack.enter_context(patch.object(Future, 'result', result))
