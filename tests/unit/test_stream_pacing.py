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
    assert all(b[3] - a[3] >= 1 / 15 - 1e-9 for a, b in zip(published, published[1:]))
    assert all(row[2] == [(1, 1, 4, 4)] for row in published[12:20])


def test_large_pts_gap_has_bounded_wait_and_stop_prevents_publication():
    clock = Clock()
    pacer = StreamPacer(clock, clock=clock)
    assert pacer.wait(Fraction(0))
    assert pacer.wait(Fraction(100))
    assert clock.now == .5
    clock.stopped = True
    assert not pacer.wait(Fraction(101))
