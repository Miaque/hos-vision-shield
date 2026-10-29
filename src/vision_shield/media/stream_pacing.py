"""Publish already safe live frames without catching up in a burst."""

import time


class StreamPacer:
    def __init__(self, stop, *, clock=time.monotonic):
        self.stop, self.clock = stop, clock
        self.previous_moment, self.previous_release = None, None
        self.previous_actual = None
        self.observer = None

    def wait(self, moment):
        due = self.clock()
        observed = dict(moment=str(moment), previous_moment=None if self.previous_moment is None else str(self.previous_moment),
            previous_release=self.previous_release, previous_actual=self.previous_actual, entered_at=due,
            interval=0., media_due=due, anti_burst_due=due, rebased=False, decision_at=due,
            remaining=0., actual_wait_seconds=0.) if self.observer else None
        if self.previous_moment is not None:
            interval = min(.5, max(0, float(moment - self.previous_moment)))
            due = max(self.previous_release + interval, self.previous_actual + interval / 2)
            checked_at = self.clock()
            decision_at = self.clock()
            remaining = due - decision_at
            if observed is not None:
                observed.update(interval=interval, media_due=self.previous_release + interval,
                    anti_burst_due=self.previous_actual + interval / 2, rebased=False,
                    rebase_checked_at=checked_at, decision_at=decision_at, remaining=remaining,
                    raw_media_interval=float(moment - self.previous_moment))
            if remaining > 0:
                started = self.clock() if observed is not None else None
                interrupted = self.stop.wait(remaining)
                if observed is not None:
                    ended = self.clock()
                    observed.update(wait_started=started, wait_ended=ended, actual_wait_seconds=ended - started)
                if interrupted:
                    if observed is not None:
                        self.observer(dict(observed, due=due, actual=self.clock(), allowed=False))
                    return False
        if self.stop.is_set():
            if observed is not None:
                self.observer(dict(observed, due=due, actual=self.clock(), allowed=False))
            return False
        # Retain the plan after stalls; the actual-release floor bounds catch-up to half an interval.
        self.previous_moment, self.previous_release, self.previous_actual = moment, due, self.clock()
        if observed is not None:
            self.observer(dict(observed, due=due, actual=self.previous_actual, allowed=True))
        return True
