"""Explicit model-task profiling must preserve pool semantics and partition critical waits."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from threading import Event, Timer

import pytest


def test_cached_pool_retains_context_task_edges_and_nonoverlapping_wait_budget(tmp_path):
    from vision_shield.detection.async_profile import AsyncProfile, instrument
    from vision_shield.detection.host_profile import scope
    trace = AsyncProfile(tmp_path)
    started, finish = Event(), Event()
    with ThreadPoolExecutor(1) as pool:
        # This pool exists before opt-in instrumentation, like a resident media worker.
        with ExitStack() as stack:
            instrument(stack, trace)
            with scope(frame=7, source_seconds=.24, purpose='discovery', view='global'):
                def work():
                    started.set()
                    with trace.span('forward-primary-plate'):
                        assert finish.wait(2)
                    return 'same-result'
                pending = pool.submit(work)
                assert started.wait(1)
                timer = Timer(.02, finish.set)
                timer.start()
                try:
                    assert pending.result() == 'same-result'
                finally:
                    timer.join()
    trace.close()
    result = trace.report()
    assert result['jobs'] == 1 and result['wait_calls'] == 1
    assert result['wait_seconds'] > 0
    assert result['wait_seconds'] == pytest.approx(sum(result[k] for k in (
        'queue_dispatch_inside_wait_seconds', 'worker_inside_wait_seconds', 'after_worker_inside_wait_seconds')))
    rows = [json.loads(line) for line in (tmp_path/'async-tasks.jsonl').read_text().splitlines()]
    names = {r['event'] for r in rows}
    assert {'submit', 'worker_start', 'worker_return', 'future_done_callback', 'result_enter', 'result_exit', 'span'} <= names
    assert all(r.get('frame') == 7 for r in rows if r['event'] in ('worker_start', 'span'))
    assert len({r['job_id'] for r in rows if r.get('job_id')}) == 1


def test_queued_job_failure_is_preserved_and_trace_restores_patch(tmp_path):
    from vision_shield.detection.async_profile import AsyncProfile, instrument
    original = ThreadPoolExecutor.submit
    trace = AsyncProfile(tmp_path)
    release = Event()
    with ThreadPoolExecutor(1) as pool:
        blocker = pool.submit(lambda: release.wait(2))
        with ExitStack() as stack:
            instrument(stack, trace)
            def fails():
                raise ValueError('original-error')
            pending = pool.submit(fails)
            timer = Timer(.02, release.set)
            timer.start()
            try:
                with pytest.raises(ValueError, match='original-error'):
                    pending.result()
            finally:
                timer.join()
            assert blocker.result()
    trace.close()
    result = trace.report()
    assert result['queue_dispatch_seconds'] > 0
    assert result['failed_jobs'] == 1
    assert ThreadPoolExecutor.submit is original


def test_profile_trace_is_opt_in_and_bounded(tmp_path):
    from vision_shield.detection.host_profile import ProfileTimes
    collector = ProfileTimes(tmp_path/'plain')
    assert collector.async_tasks is None
    collector.close()
    assert not (tmp_path/'plain/async-tasks.jsonl').exists()
    from vision_shield.detection.async_profile import AsyncProfile
    trace = AsyncProfile(tmp_path/'bounded', byte_limit=1)
    with pytest.raises(RuntimeError, match='budget'):
        trace.event('example')
    trace.close()
    with pytest.raises(RuntimeError, match='budget'):
        trace.report()


def test_context_pool_correlates_nested_tasks_without_changing_callable_named_run(tmp_path):
    from vision_shield.detection.async_profile import AsyncProfile, JOB, instrument
    from vision_shield.detection.host_profile import ContextPool, CONTEXT, scope
    trace = AsyncProfile(tmp_path)
    with ContextPool(1) as pool, ThreadPoolExecutor(1) as nested, ExitStack() as stack:
        instrument(stack, trace)
        def work():
            assert JOB.get() is not None and CONTEXT.get()['frame'] == 9
            parent = JOB.get()
            return parent, nested.submit(lambda: (JOB.get(), CONTEXT.get()['frame'])).result()
        # An ordinary function with this name must not be mistaken for Context.run.
        def run(fn):
            return ('ordinary', fn())
        with scope(frame=9):
            parent, (child, frame) = pool.submit(work).result()
            assert frame == 9 and child != parent
            assert pool.submit(run, lambda: 'argument').result() == ('ordinary', 'argument')
    trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert next(r for r in rows if r['event']=='submit' and r['job_id']==child)['parent_job'] == parent
    assert trace.report()['clock']['monotonic'] is True


def test_model_observer_restores_resident_methods_after_diagnostic(tmp_path):
    from types import SimpleNamespace
    from vision_shield.detection.host_profile import ProfileTimes, instrument_candidates
    class Model:
        def candidates(self, pixels, *, kind=None):
            return []
        def _views(self, width, height, side):
            yield 0, 0, width, height
    model = Model()
    original = model.candidates
    trace = ProfileTimes(tmp_path, async_tasks=True)
    with ExitStack() as stack:
        instrument_candidates(model, 'primary', trace, stack=stack)
        assert model.candidates(SimpleNamespace(shape=(480,640,3)), kind='plate') == []
    trace.close()
    assert model.candidates == original
    assert model.candidates(None) == []
    assert 'candidates' not in model.__dict__ and '_views' not in model.__dict__


def test_resident_child_collects_existing_pools_and_restores_next_plain_request(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from vision_shield.config import Settings
    from vision_shield.detection import host_profile
    from vision_shield.detection.timing import StageTimes
    from vision_shield.media import child, video
    class Model:
        def __init__(self, group, pool):
            self.group, self.pool, self.stage_times = group, pool, StageTimes()
        def candidates(self, pixels, **kwargs):
            def forward():
                with self.stage_times.measure(f'forward-{self.group}-plate'):
                    return []
            return self.pool.submit(forward).result()
    def process_video(source, target, detector, cfg, stop, emit):
        for model in (detector.primary, detector.detail):
            assert model.candidates(SimpleNamespace(shape=(320,320,3)),kind='plate') == []
        Path(target).write_bytes(b'fixture-output')
    monkeypatch.setattr(video, 'process_video', process_video)
    monkeypatch.setattr(host_profile, 'runtime_metadata', lambda *a: {})
    events = []
    monkeypatch.setattr(child, 'emit', events.append)
    cfg = Settings(_env_file=None, debug=True, detection_backend='quality_yolo', inference_device='cuda').model_dump()
    with ThreadPoolExecutor(1) as pool:
        detector = SimpleNamespace(stage_times=StageTimes(), primary=Model('primary',pool),detail=Model('detail',pool))
        originals = [owner.stage_times for owner in (detector,detector.primary,detector.detail)]
        payload = dict(kind='video',source='fixture',directory=str(tmp_path/'profiled'),config=cfg,profile_detection='async-host')
        assert child.process(payload, Event(), detector) == (0,True)
        report = next(e['device_profile'] for e in events if 'device_profile' in e)
        assert report['profile']['async_tasks']['jobs'] == 2
        assert report['profile']['models']['forward-primary-plate']['calls'] == 1
        assert report['profile']['models']['forward-detail-plate']['calls'] == 1
        assert [owner.stage_times for owner in (detector,detector.primary,detector.detail)] == originals
        del payload['profile_detection']
        payload['directory'] = str(tmp_path/'plain')
        assert child.process(payload, Event(), detector) == (0,True)
        assert not (tmp_path/'plain/host-profile').exists()
