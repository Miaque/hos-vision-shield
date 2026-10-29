"""Publish already safe live frames without catching up in a burst."""

import time


class StreamPacer:
    def __init__(self, stop, *, clock=time.monotonic):
        self.stop, self.clock = stop, clock
        self.previous_moment, self.previous_release = None, None
        self.previous_actual = None

    def wait(self, moment):
        due = self.clock()
        if self.previous_moment is not None:
            interval = min(.5, max(0, float(moment - self.previous_moment)))
            due = max(self.previous_release + interval, self.previous_actual + interval / 2)
            if due < self.clock() - interval / 2:
                due = self.clock()
            remaining = due - self.clock()
            if remaining > 0 and self.stop.wait(remaining):
                return False
        if self.stop.is_set():
            return False
        # Keep ordinary timer jitter off the timeline; a stall rebases instead of accumulating credit.
        self.previous_moment, self.previous_release, self.previous_actual = moment, due, self.clock()
        return True
