"""Local scheduling experiment, with the real pacer and lookahead; no media I/O."""
from fractions import Fraction
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

from vision_shield.media import plate_lookahead
from vision_shield.media.plate_lookahead import PendingMask, PlateLookahead
from vision_shield.media.stream_pacing import StreamPacer


class Clock:
    def __init__(self):
        self.now, self.waited, self.stopped = 0., 0., False

    def __call__(self):
        return self.now

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        assert 0 < seconds <= .5 + 1e-8
        self.now += seconds
        self.waited += seconds
        return self.stopped


def without_late_rebase():
    # Only remove this exact branch, retaining the half-interval floor and .5s cap.
    original = inspect.getsource(StreamPacer)
    branch = '            if due < self.clock() - interval / 2:\n                due = self.clock()\n'
    assert original.count(branch) == 1, 'Re-audit experiment when the baseline changes'
    namespace = {}
    exec(original.replace(branch, ''), {'time': SimpleNamespace(monotonic=lambda: 0)}, namespace)
    return namespace['StreamPacer']


def simulate(costs, pacer_type, count=1200, *, separated=False):
    clock = Clock()
    owner = SimpleNamespace(debug=True, detection_stats={})
    delay = PlateLookahead(owner)
    publish_clock = Clock() if separated else clock
    pacer = pacer_type(publish_clock, clock=publish_clock)
    outputs, decisions, batches = [], [], []
    queued_starts, max_queued = [], 0

    def release(ready):
        nonlocal max_queued
        batches.append(len(ready))
        for index, pts, masks, rendered in ready:
            if separated:
                # Discrete-event single writer, four WAITING final frames plus one active.
                queued_starts[:] = [t for t in queued_starts if t > clock.now + 1e-9]
                if len(queued_starts) == 4:
                    clock.now = max(clock.now, queued_starts.pop(0))
                start = max(publish_clock.now, clock.now)
                if start > clock.now + 1e-9:
                    queued_starts.append(start)
                max_queued = max(max_queued, len(queued_starts))
                assert len(queued_starts) <= 4
                publish_clock.now = start
            publish_clock.now += .001  # Same 1ms encode/mux model in both arms.
            now, waited = publish_clock.now, publish_clock.waited
            epoch = outputs[0]['wall'] if outputs else now
            assert pacer.wait(pts)
            decisions.append(dict(sequence=index + 1, source_pts=str(pts), now=now,
                due=pacer.previous_release, actual=publish_clock.now, wait=publish_clock.waited - waited,
                fixed_epoch_due=epoch + float(pts), rendered=rendered,
                late_to_fixed_epoch=now - (epoch + float(pts))))
            outputs.append(dict(index=index, pts=pts, masks=masks, wall=publish_clock.now))

    with patch.object(plate_lookahead, 'time', SimpleNamespace(monotonic=clock)):
        for index in range(count):
            release(delay.expire())  # Match the actual main-loop ordering.
            pts = Fraction(index, 25)
            clock.now = max(clock.now, float(pts))
            clock.now += costs[index % len(costs)]
            pending = PendingMask(float(pts), np.zeros((8, 8), np.uint8), [], [],
                lambda extra, index=index, pts=pts: (index, pts, tuple(extra), clock.now),
                deadline=clock.now + .75)
            release(delay.push(pending))
            assert len(delay.pending) <= 16
        release(delay.flush())
    assert [o['index'] for o in outputs] == list(range(count))
    assert [o['pts'] for o in outputs] == [Fraction(i, 25) for i in range(count)]
    intervals = [b['wall'] - a['wall'] for a, b in zip(outputs, outputs[1:])]
    assert min(intervals) >= .02 - 1e-8  # Bounded catch-up, never an unrestricted burst.
    elapsed = outputs[-1]['wall'] - outputs[0]['wall']
    late = [d for d in decisions if d['late_to_fixed_epoch'] > .04 and d['wait'] > 1e-8]
    return dict(frames=count, wall_seconds=elapsed, media_seconds=(count - 1) / 25,
        fps=(count - 1) / elapsed, lag_growth_per_second=1 - (count - 1) / 25 / elapsed,
        pacing_seconds=publish_clock.waited, late_positive_wait_count=len(late),
        late_positive_wait_seconds=sum(d['wait'] for d in late),
        min_interval=min(intervals), max_interval=max(intervals),
        max_release_batch=max(batches), mean_compute_seconds=sum(costs)/len(costs),
        max_final_queue_frames=max_queued,
        max_final_ready_to_publish_seconds=max(d['actual']-d['rendered'] for d in decisions),
        decisions=decisions)


def self_check(candidate):
    # The real method already does NOT sleep when late to its current due.
    clock = Clock()
    pacer = StreamPacer(clock, clock=clock)
    pacer.wait(Fraction(0))
    clock.now = 1
    pacer.wait(Fraction(1, 25))
    assert clock.waited == 0
    clock.now += .005
    pacer.wait(Fraction(2, 25))
    assert abs(clock.waited - .035) < 1e-8
    # Both policies keep timer jitter bounded and stop requests interrupt publication.
    for cls in (StreamPacer, candidate):
        clock = Clock()
        pacer = cls(clock, clock=clock)
        for i in range(300):
            assert pacer.wait(Fraction(i, 25))
        assert abs(clock.now - 299 / 25) < 1e-8
        assert pacer.wait(Fraction(100)) and clock.waited <= 299 / 25 + .5 + 1e-8
        clock.stopped = True
        assert not pacer.wait(Fraction(101))


def main():
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    candidate = without_late_rebase()
    self_check(candidate)
    result = {}
    for name, costs in [('light', [.005]), ('jitter_below_budget', [.09, .01, .01]),
                        ('compute_above_budget', [.1, .025, .025])]:
        result[name] = {}
        for policy, cls in [('baseline', StreamPacer), ('keep_planned_due', candidate),
                            ('separate_writer_baseline_pacer', StreamPacer)]:
            arm = simulate(costs, cls, separated=policy.startswith('separate'))
            (folder / f'{name}-{policy}.json').write_text(json.dumps(arm), encoding='utf-8')
            del arm['decisions']
            result[name][policy] = arm
        assert result[name]['keep_planned_due']['wall_seconds'] <= result[name]['baseline']['wall_seconds'] + 1e-8
    (folder / 'counterfactual-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
