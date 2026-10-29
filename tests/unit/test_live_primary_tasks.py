"""Correlated live waits partition one critical path without adding concurrent work."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, Timer

import pytest

from vision_shield.media.stream_e0 import StreamE0


def test_live_primary_wait_partitions_forward_and_preserves_result_and_failure():
    trace, started, release = StreamE0(), Event(), Event()
    def model(value, **kwargs):
        with trace.span('stage:prepare-primary'):
            pass
        with trace.span('stage:forward-primary-plate'):
            started.set()
            assert release.wait(2)
        with trace.span('stage:post-primary'):
            if value is None:
                raise ValueError('unsafe')
            return value
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = trace.submit_primary(pool, model, 42, kind='plate')
        assert started.wait(2)
        timer = Timer(.03, release.set)
        timer.start()
        try:
            assert trace.primary_result(future) == 42
        finally:
            release.set(); timer.join()
        with pytest.raises(ValueError, match='unsafe'):
            trace.primary_result(trace.submit_primary(pool, model, None, kind='plate'))
    state = trace.snapshot()
    totals = state['primary_task_totals']
    assert totals['submitted'] == totals['completed'] == totals['wait_calls'] == 2
    assert totals['failed'] == 1
    assert totals['wait_stage:forward-primary-plate_seconds'] > .01
    assert sum(totals[key] for key in ('wait_queue_dispatch_seconds', 'wait_worker_seconds',
                                      'wait_after_worker_seconds')) == pytest.approx(totals['wait_seconds'])
    assert sum(value for key,value in totals.items() if key.startswith('wait_stage:')) <= totals['wait_worker_seconds']+1e-6
    assert state['primary_tasks'][0]['kind'] == 'plate'
    assert state['primary_tasks'][0]['worker_return'] <= state['primary_tasks'][0]['result_exit']


def test_primary_nested_spans_have_disjoint_wait_intersections():
    from vision_shield.media.stream_e0 import primary_segments
    spans = [dict(stage='stage:outer',started=1.,ended=5.,cpu=2.),
             dict(stage='stage:inner',started=2.,ended=4.,cpu=1.)]
    segments = primary_segments(spans)
    assert [(a,b,name) for a,b,name in segments] == [
        (1.,2.,'stage:outer'),(4.,5.,'stage:outer'),(2.,4.,'stage:inner')]
    assert sum(b-a for a,b,_ in segments) == 4


def test_live_primary_trace_eviction_keeps_all_counts_and_wire_is_bounded():
    trace = StreamE0()
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i in range(150):
            assert trace.primary_result(trace.submit_primary(pool, lambda x:x, i)) == i
    state = trace.snapshot()
    assert state['primary_task_totals']['completed'] == 150
    assert state['discarded']['primary_tasks'] == 150-len(state['primary_tasks'])
    assert state['ring_discarded']['primary_tasks'] > 0
    assert len(json.dumps(state).encode()) <= 128*1024


def test_primary_hooks_are_only_enabled_with_e0_and_restored():
    from types import SimpleNamespace
    from vision_shield.detection.timing import StageTimes
    from vision_shield.media.live_diagnostics import LiveDiagnostics
    timing = StageTimes()
    owner = SimpleNamespace(stage_times=timing)
    for enabled in (False, True):
        diagnostics = LiveDiagnostics(owner, {'stream_e0_diagnostics':enabled}, 'a', runtime={})
        assert timing.primary_observer is (diagnostics.e0 if enabled else None)
        diagnostics.close()
        assert timing.primary_observer is None


@pytest.mark.parametrize('cached', [False,True])
@pytest.mark.parametrize('failure', [None,'primary','detail'])
def test_live_observer_preserves_both_quality_routes_inputs_results_and_failures(cached, failure):
    import numpy as np
    from vision_shield.detection import Detection
    from vision_shield.detection.quality import Detector
    from vision_shield.detection.timing import StageTimes, measure
    from vision_shield.media.live_diagnostics import LiveDiagnostics
    from collections import Counter
    outcomes=[]
    for enabled in (False,True):
        calls=[]
        face=Detection('face',(1,1,4,4),.9)
        class Model:
            def __init__(self,name): self.name=name
            def prepare(self,pixels): return pixels
            def proposals(self,pixels): return self.candidates(pixels,kind='face')
            def candidates(self,pixels,*,kind=None):
                with measure(self,'forward-primary-face' if self.name=='primary' else 'forward-detail-plate'):
                    assert not pixels.flags.writeable
                    calls.append((self.name,kind))
                    if self.name==failure: raise ValueError(self.name)
                    return [face] if self.name=='primary' else []
        owner=Detector.__new__(Detector)
        owner.inference_device,owner.box_padding='cuda',0
        owner.primary,owner.detail=Model('primary'),Model('detail')
        owner.stage_times=owner.primary.stage_times=owner.detail.stage_times=StageTimes()
        diagnostics=LiveDiagnostics(owner,{'stream_e0_diagnostics':enabled},'a',runtime={})
        pixels=np.full((8,8,3),180,np.uint8); pixels.setflags(write=False)
        try:
            with diagnostics.operation('process','1/25'):
                if cached: owner.plate_proposals(pixels)
                result=owner.detect_faces(pixels) if cached else owner.detect_regions(pixels)
            outcome=result
        except ValueError as error:
            outcome=str(error)
        finally:
            owner._pool.shutdown();diagnostics.close()
        assert (pixels==180).all()
        outcomes.append((outcome,Counter(calls)))
        if enabled:
            assert diagnostics.e0.snapshot()['primary_task_totals']['completed']==1
    assert outcomes[0]==outcomes[1]


def test_overflow_is_reported_without_misattributing_incomplete_nested_phases():
    trace=StreamE0()
    def work():
        with trace.span('stage:forward-primary-plate'):
            for _ in range(150):
                with trace.span('stage:prepare-primary'): pass
    with ThreadPoolExecutor(max_workers=1) as pool:
        trace.primary_result(trace.submit_primary(pool,work))
    totals=trace.snapshot()['primary_task_totals']
    assert totals['spans_discarded']>0
    assert totals['wait_unattributed_worker_seconds']==totals['wait_worker_seconds']
    assert not any(k.startswith('wait_stage:') for k in totals)


def test_primary_snapshot_is_not_mutated_by_a_late_completion_callback():
    trace=StreamE0()
    with ThreadPoolExecutor(max_workers=1) as pool:
        trace.primary_result(trace.submit_primary(pool,lambda:42))
    before=trace.snapshot()['primary_tasks'][0]
    previous=before['revision']
    with trace.lock:
        trace.primary_tasks[0]['revision']+=1
    assert before['revision']==previous


def test_live_primary_analysis_uses_completed_counters_when_every_task_record_is_evicted(monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'specs/007-model-async/experiments/analyze-live-primary.py'
    spec=importlib.util.spec_from_file_location('live_primary_analysis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'verify_runtime',lambda *args:None)
    first=dict(started_monotonic=0.,sampled_monotonic=1.,ended_monotonic=None,
               primary_task_totals=dict(submitted=1,completed=1,wait_seconds=1.,wait_worker_seconds=1.,
                                        **{'wait_stage:forward-primary-plate_seconds':1.}))
    last=dict(first,sampled_monotonic=3.,primary_task_totals=dict(submitted=4,completed=4,
        wait_seconds=3.,wait_queue_dispatch_seconds=.2,wait_worker_seconds=2.5,
        wait_after_worker_seconds=.3,wait_unattributed_worker_seconds=.5,
        **{'wait_stage:forward-primary-plate_seconds':2.}))
    rows=[dict(kind='runtime',data={}),dict(kind='e0_state',attempt_id='a',data=first),
          dict(kind='e0_state',attempt_id='a',data=last)]
    result=module.analyze(rows,'ref')['attempts']['a']
    assert result['totals']['completed']==3
    assert result['totals']['wait_seconds']==2.
    assert result['missing_task_records']==3 and result['retained_task_records']==0


def test_queued_primary_is_separate_from_execution_and_already_done_is_not_added_to_wait():
    trace,blocked,release=StreamE0(),Event(),Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        def occupy():
            blocked.set(); assert release.wait(2)
        occupied=pool.submit(occupy)
        assert blocked.wait(2)
        future=trace.submit_primary(pool,lambda:42)
        timer=Timer(.03,release.set);timer.start()
        try: assert trace.primary_result(future)==42
        finally: release.set();timer.join();occupied.result()
        job=trace.snapshot()['primary_tasks'][-1]
        assert job['partition']['wait_queue_dispatch_seconds']>.01
        complete=trace.submit_primary(pool,lambda:43)
        assert complete.result()==43
        assert trace.primary_result(complete)==43
        done=trace.snapshot()['primary_tasks'][-1]
        assert done['partition']['wait_worker_seconds']==0
        assert done['partition']['wait_queue_dispatch_seconds']==0
