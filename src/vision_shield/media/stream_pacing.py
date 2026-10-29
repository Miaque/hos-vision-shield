"""Publish already safe live frames without catching up in a burst."""

import time


class StreamPacer:
    def __init__(self, stop, *, clock=time.monotonic):
        self.stop, self.clock = stop, clock
        self.previous_moment, self.previous_release = None, None

    def wait(self, moment):
        if self.previous_moment is not None:
            interval = min(.5, max(0, float(moment - self.previous_moment)))
            remaining = self.previous_release + interval - self.clock()
            if remaining > 0 and self.stop.wait(remaining):
                return False
        if self.stop.is_set():
            return False
        # Rebase after a stall: a late frame never gives the next frame a catch-up credit.
        self.previous_moment, self.previous_release = moment, self.clock()
        return True
