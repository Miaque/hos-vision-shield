"""E0 observes the existing media decisions, including decisions evicted from its rings."""
from fractions import Fraction
import json
import threading
from types import SimpleNamespace

import pytest

from vision_shield.media.stream_pacing import StreamPacer


class Clock:
    def __init__(self):
        self.now, self.waits, self.stopped = 0., [], False

    def __call__(self):
        return self.now

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        return self.stopped


def test_pacer_observation_keeps_e1_decisions_without_late_rebase_contribution():
    from vision_shield.media.stream_e0 import StreamE0
    results = []
    for observed in (False, True):
        clock = Clock()
        trace = StreamE0(clock=clock) if observed else None
        pacer = StreamPacer(clock, clock=clock)
        pacer.observer = trace.pacing if trace else None
        for index, ready_at in enumerate([0, 1, 1.005, 1.045]):
            clock.now = max(clock.now, ready_at)
            assert pacer.wait(Fraction(index, 25))
        results.append((clock.now, clock.waits))
    assert results[0] == results[1]
    state = trace.snapshot()
    assert state['pacer_totals']['decisions'] == 4
    assert state['pacer_totals']['rebases'] == 0
    assert state['pacer_totals']['conditional_rebase_request_seconds'] == pytest.approx(0)
    record = state['pacer'][-2]
    assert record['due'] == pytest.approx(1.02)
    assert record['remaining'] == pytest.approx(.015)
    assert record['actual_wait_seconds'] == pytest.approx(.015)
    assert record['conditional_rebase_request_seconds'] == pytest.approx(0)
    assert record['retained_anti_burst_request_seconds'] == pytest.approx(.015)


def test_pacer_counters_cover_eviction_and_wait_partition_is_complete():
    from vision_shield.media.stream_e0 import StreamE0
    clock, trace = Clock(), StreamE0(clock=lambda: 0.)
    pacer = StreamPacer(clock, clock=clock)
    pacer.observer = trace.pacing
    for i in range(2000):
        assert pacer.wait(Fraction(i, 25))
    state = trace.snapshot()
    totals = state['pacer_totals']
    assert totals['decisions'] == 2000
    assert state['discarded']['pacer'] == 2000 - len(state['pacer'])
    assert totals['actual_wait_seconds'] == pytest.approx(sum(clock.waits))
    assert sum(totals[key] for key in ('conditional_rebase_wait_seconds',
        'retained_anti_burst_wait_seconds', 'retained_media_wait_seconds', 'timer_overrun_seconds')) == pytest.approx(sum(clock.waits))
    assert totals['conditional_rebase_request_seconds'] == pytest.approx(0)
    assert len(json.dumps(state).encode()) <= 128 * 1024


def test_exclusive_stage_budget_subtracts_nested_intervals_and_separates_threads():
    from vision_shield.media.stream_e0 import StreamE0
    clock = Clock()
    trace = StreamE0(clock=clock)
    with trace.span('op:process'):
        clock.now = 1
        with trace.span('stage:track-recheck'):
            clock.now = 2
            with trace.span('stage:explicit-sync'):
                clock.now = 2.5
                partial = trace.snapshot()
                assert sum(partial['main_exclusive_seconds'].values()) == 2.5
                assert partial['main_unattributed_seconds'] == 0
                clock.now = 3
            clock.now = 4
        clock.now = 5
    state = trace.snapshot()
    budget = state['main_exclusive_seconds']
    assert budget == {'op:process': 2, 'stage:track-recheck': 2, 'stage:explicit-sync': 1}
    assert state['main_unattributed_seconds'] == 0
    with pytest.raises(ValueError):
        with trace.span('op:failure'):
            clock.now = 6
            raise ValueError('test')
    assert trace.snapshot()['main_exclusive_seconds']['op:failure'] == 1
    with trace.span('op:process'):
        def parallel():
            with trace.span('stage:forward-primary-plate'):
                clock.now = 8
        worker = threading.Thread(target=parallel)
        worker.start()
        worker.join()
        clock.now = 9
    state = trace.snapshot()
    assert state['main_exclusive_seconds']['op:process'] == 5
    assert sum(row['stage:forward-primary-plate'] for row in state['other_thread_exclusive_seconds'].values()) == 2
    assert sum(state['main_exclusive_seconds'].values()) == 9


def test_fifo_trace_keeps_identity_raw_clock_and_full_backpressure():
    from vision_shield.media.stream_e0 import StreamE0
    from vision_shield.media.streams import StreamBuffer
    trace = StreamE0()
    buffer = StreamBuffer(1, 16, 4194304, threading.Event())
    buffer.trace = trace
    def frame(pts):
        value = SimpleNamespace(pts=pts, time_base=Fraction(1, 90000))
        fid = trace.frame_start(value, None)
        return value, fid
    a, aid = frame(9000)
    b, bid = frame(12600)
    buffer.frame(a, trace_id=aid)
    entered = threading.Event()
    def enqueue():
        entered.set()
        buffer.frame(b, trace_id=bid)
    worker = threading.Thread(target=enqueue)
    worker.start()
    assert entered.wait(1)
    # Read while the producer is waiting; querying must not depend on enqueue completing.
    for _ in range(10000):
        if trace.snapshot()['fifo_totals'].get('full_events', 0):
            break
    assert trace.snapshot()['fifo_totals']['full_events'] == 1
    assert buffer.take()[0] is a and buffer.last_trace_id == aid
    worker.join(1)
    assert not worker.is_alive()
    assert buffer.take()[0] is b and buffer.last_trace_id == bid
    state = trace.snapshot()
    assert state['fifo_totals']['enqueued'] == state['fifo_totals']['dequeued'] == 2
    assert state['fifo_totals']['backpressure_seconds'] >= 0
    row = next(row for row in state['frames'] if row['frame_id'] == bid)
    assert row['raw_frame_clock']['pts'] == 12600
    assert row['raw_frame_clock']['time_base'] == '1/90000'
    assert row['enqueue_monotonic'] <= row['dequeue_monotonic']


def test_worker_clock_anomalies_are_counted_before_repair_even_after_eviction():
    from vision_shield.media.stream_e0 import StreamE0
    trace = StreamE0()
    for i in range(1000):
        packet = SimpleNamespace(pts=i % 2, dts=i % 2, time_base=Fraction(1, 8000), duration=1024)
        before = trace.media_clock(packet)
        packet.pts = packet.dts = i * 1024
        trace.audio_clock(before, trace.media_clock(packet))
    state = trace.snapshot()
    assert state['clock_totals']['audio_packets'] == 1000
    assert state['clock_totals']['audio_raw_dts_nonincreasing'] == 499
    assert state['clock_totals'].get('audio_normalized_dts_nonincreasing', 0) == 0
    assert state['clock_origin'] == 'worker_pyav_demux_and_decode'


@pytest.mark.parametrize('mode', ['stop', 'oversleep', 'large-gap'])
def test_wait_accounting_keeps_stop_timer_overrun_and_gap_cap_separate(mode):
    from vision_shield.media.stream_e0 import StreamE0
    class Timer(Clock):
        def wait(self, seconds):
            self.waits.append(seconds)
            self.now += seconds / 4 if mode == 'stop' else seconds + .001 if mode == 'oversleep' else seconds
            self.stopped = mode == 'stop'
            return self.stopped
    clock = Timer()
    trace = StreamE0(clock=clock)
    pacer = StreamPacer(clock, clock=clock)
    pacer.observer = trace.pacing
    assert pacer.wait(Fraction(0))
    with trace.span('op:output-pacing'):
        result = pacer.wait(Fraction(100) if mode == 'large-gap' else Fraction(1, 25))
    totals = trace.snapshot()['pacer_totals']
    assert result is (mode != 'stop')
    assert totals['decisions'] == 2
    assert totals['stopped_decisions'] == (mode == 'stop')
    assert totals['timer_overrun_seconds'] == pytest.approx(.001 if mode == 'oversleep' else 0)
    assert totals['clamped_media_intervals'] == (mode == 'large-gap')
    assert trace.snapshot()['main_exclusive_seconds']['wait:pacer'] == pytest.approx(totals['actual_wait_seconds'])
    assert sum(totals[key] for key in ('conditional_rebase_wait_seconds', 'retained_anti_burst_wait_seconds',
        'retained_media_wait_seconds', 'timer_overrun_seconds')) == pytest.approx(totals['actual_wait_seconds'])


def test_e0_can_be_disabled_without_replacing_existing_observers():
    from vision_shield.detection.timing import StageTimes
    from vision_shield.media.live_diagnostics import LiveDiagnostics
    times = StageTimes()
    original = times.span_observer
    trace = LiveDiagnostics(SimpleNamespace(stage_times=times), {'stream_e0_diagnostics': False}, 'a', runtime={})
    assert trace.e0 is None and times.span_observer is original
    assert 'e0' not in trace.snapshot()
    trace.close()


def test_observed_e1_pacer_keeps_bounds_and_does_not_add_wait_to_fixed_baseline():
    import random
    import subprocess
    from vision_shield.media.stream_e0 import StreamE0
    namespace = {}
    baseline = subprocess.check_output(['git', 'show',
        '1c9be543c30fed7a2f4263b85c447ba079d438f8:src/vision_shield/media/stream_pacing.py'], text=True)
    exec(baseline, namespace)
    rng = random.Random(329)
    moments, work, moment = [], [], Fraction(0)
    for _ in range(1000):
        moment += rng.choice([Fraction(1, 25), Fraction(1, 25), Fraction(1, 90000), Fraction(3), -Fraction(1, 25)])
        moments.append(moment)
        work.append(rng.choice([.002, .01, .05, .12, 1.2]))
    results = []
    for cls, observed in [(namespace['StreamPacer'], False), (StreamPacer, False), (StreamPacer, True)]:
        clock = Clock()
        pacer = cls(clock, clock=clock)
        if observed:
            trace = StreamE0(clock=clock)
            pacer.observer = trace.pacing
        rows = []
        for pts, seconds in zip(moments, work):
            clock.now += seconds
            assert pacer.wait(pts)
            rows.append((pacer.previous_release, pacer.previous_actual))
        results.append((rows, clock.waits))
    assert results[1] == results[2]
    assert sum(results[1][1]) <= sum(results[0][1])
    assert all(actual >= due for due, actual in results[1][0])
