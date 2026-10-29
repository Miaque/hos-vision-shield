from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from vision_shield.media.plate_lookahead import PendingMask, PlateLookahead
from vision_shield.media.stream_pacing import StreamPacer


class Clock:
    now = 0.0
    stopped = False

    def __call__(self):
        return self.now

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        assert 0 < seconds <= .5 + 1e-9
        self.now += seconds
        return self.stopped


@pytest.mark.parametrize('stall', [0, 2])
def test_zero_model_scene_cut_spaces_all_eight_ready_frames_without_rewriting_clocks(stall):
    clock = Clock()
    owner = SimpleNamespace(debug=False, detection_stats={})
    lookahead = PlateLookahead(owner)
    pacer = StreamPacer(clock, clock=clock)
    published, batches = [], []

    def release(ready):
        batches.append(len(ready))
        for index, pts, masks in ready:
            assert pacer.wait(pts)
            published.append((index, pts, masks, clock()))

    for index in range(40):
        clock.now = max(clock.now, index / 15)
        if index == 20:
            clock.now += stall
        pts = Fraction(index, 15)
        pending = PendingMask(float(pts), np.zeros((8, 8), np.uint8), [], [],
                              lambda extra, index=index, pts=pts: (index, pts, list(extra)), reset=index == 20,
                              discover=lambda: [(1, 1, 4, 4)])
        release(lookahead.push(pending))
    release(lookahead.flush())
    assert max(batches[:-1]) == 8  # Actual scene-reset flush, with zero model cost.
    assert [row[0] for row in published] == list(range(40))
    assert [row[1] for row in published] == [Fraction(i, 15) for i in range(40)]
    # E1 retains a half-interval floor after stalls instead of rebasing to a full interval.
    minimum = 1 / 30 if stall else 1 / 15
    assert all(b[3] - a[3] >= minimum - 1e-9 for a, b in zip(published, published[1:]))
    assert all(row[2] == [(1, 1, 4, 4)] for row in published[12:20])


def test_large_pts_gap_has_bounded_wait_and_stop_prevents_publication():
    clock = Clock()
    pacer = StreamPacer(clock, clock=clock)
    assert pacer.wait(Fraction(0))
    assert pacer.wait(Fraction(100))
    assert clock.now == .5
    clock.stopped = True
    assert not pacer.wait(Fraction(101))


def test_normal_timer_oversleep_does_not_accumulate_latency():
    class LateClock(Clock):
        def wait(self, seconds):
            self.now += seconds + .001
            return False

    clock = LateClock()
    pacer = StreamPacer(clock, clock=clock)
    released = []
    for i in range(300):
        assert pacer.wait(Fraction(i, 15))
        released.append(clock.now)
    assert released[-1] - 299 / 15 < .002
    assert min(b - a for a, b in zip(released, released[1:])) >= 1 / 30 - 1e-9


def test_late_release_keeps_planned_due_and_bounds_subsequent_catchup():
    clock = Clock()
    pacer = StreamPacer(clock, clock=clock)
    assert pacer.wait(Fraction(0))
    clock.now = 1.
    assert pacer.wait(Fraction(1, 25))
    # The late frame itself does not wait or advance the media clock.
    assert clock.now == 1.
    assert pacer.previous_release == .04
    clock.now += .005
    assert pacer.wait(Fraction(2, 25))
    assert clock.now == pytest.approx(1.02)
    assert pacer.previous_moment == Fraction(2, 25)


def test_stop_interrupts_an_in_progress_large_gap_wait():
    from threading import Event, Thread
    from time import perf_counter
    stop, entered, results = Event(), Event(), []
    wait = stop.wait
    def observed_wait(seconds):
        entered.set()
        return wait(seconds)
    stop.wait = observed_wait
    pacer = StreamPacer(stop, clock=perf_counter)
    assert pacer.wait(Fraction(0))
    worker = Thread(target=lambda: results.append(pacer.wait(Fraction(100))))
    worker.start()
    assert entered.wait(1)
    stop.set()
    worker.join(.5)
    assert not worker.is_alive() and results == [False]
    assert pacer.previous_moment == Fraction(0)


def test_late_frame_floor_does_not_rebase_the_following_media_plan():
    clock = Clock()
    pacer = StreamPacer(clock, clock=clock)
    observed = []
    pacer.observer = observed.append
    assert pacer.wait(Fraction(0))
    # The live 336-338 case has this shape: late frame, floor, then another fast frame.
    clock.now = .110
    assert pacer.wait(Fraction(1, 25))
    clock.now = .111
    assert pacer.wait(Fraction(2, 25))
    assert clock.now == pytest.approx(.130, rel=0, abs=1e-12)
    clock.now = .131
    assert pacer.wait(Fraction(3, 25))
    assert clock.now == pytest.approx(.150, rel=0, abs=1e-12)
    assert pacer.previous_release == pytest.approx(.120, rel=0, abs=1e-12)
    assert [row['moment'] for row in observed] == ['0', '1/25', '2/25', '3/25']
    assert all(b['actual']-a['actual'] >= .020-1e-12 for a, b in zip(observed, observed[1:]))


@pytest.mark.parametrize('step', [Fraction(5, 90000), Fraction(6, 90000), Fraction(1, 25), Fraction(1)])
def test_independent_media_plan_retains_each_interval_floor_and_stop_state(step):
    class InterruptibleClock(Clock):
        interrupt = False

        def wait(self, seconds):
            if self.interrupt:
                self.now += seconds / 2
                self.stopped = True
                return True
            return super().wait(seconds)

    clock = InterruptibleClock()
    pacer = StreamPacer(clock, clock=clock)
    assert pacer.wait(Fraction(0))
    clock.now = 1.
    assert pacer.wait(step)
    previous = clock.now
    assert pacer.wait(2*step)
    assert clock.now-previous >= min(.5, float(step))/2-1e-12
    assert pacer.previous_release == pytest.approx(2*min(.5, float(step)), rel=0, abs=1e-12)
    before = pacer.previous_moment, pacer.previous_release, pacer.previous_actual
    clock.interrupt = True
    assert not pacer.wait(3*step)
    assert before == (pacer.previous_moment, pacer.previous_release, pacer.previous_actual)
