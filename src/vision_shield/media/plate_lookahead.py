"""Bounded video lookahead: confirmed new plates can mask earlier matched pictures."""

from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable
from contextlib import nullcontext
import time

import numpy as np


@dataclass
class PendingMask:
    moment: float
    gray: np.ndarray
    plates: list
    confirmed: list
    render: Callable
    reset: bool = False
    confirm: Callable | None = None
    discover: Callable | None = None
    extra: list = field(default_factory=list)
    deadline: float | None = None
    trace_id: int | None = None


class PlateLookahead:
    def __init__(self, redactor):
        self.redactor, self.pending, self.previous = redactor, deque(), []
        self.trace = None
        redactor._plate_template_fallback = True
        redactor.detection_stats.update(backfilled_plate_masks=0, lookahead_max_frames=0,
                                       backfill_local_checks=0, lookahead_tail_checks=0)

    def push(self, frame):
        ready = self.flush(reason='scene-reset') if frame.reset else self.expire()
        if frame.reset:
            self.previous = []
        for box in frame.confirmed:
            if any(self.redactor._overlap(box, old) >= .2 for old in self.previous):
                continue
            reference = frame.gray
            for old in reversed(self.pending):
                if frame.moment - old.moment > .5:
                    break
                recovered = self.redactor._follow(old.gray, box, reference)
                if recovered is None and old.confirm is not None:
                    if self.redactor.debug:
                        self.redactor.detection_stats['backfill_local_checks'] += 1
                    recovered = old.confirm(box)
                if recovered is None:
                    break
                box = recovered
                old.extra.append(box)
                reference = old.gray
                if self.redactor.debug:
                    self.redactor.detection_stats['backfilled_plate_masks'] += 1
        self.previous = frame.plates
        self.pending.append(frame)
        if self.trace:
            self.trace.update_frame(frame.trace_id, lookahead_entered_monotonic=time.monotonic(),
                media_deadline=frame.moment + .5, wall_deadline=frame.deadline,
                lookahead_depth_at_entry=len(self.pending))
        while (len(self.pending) > 16 or self.pending
               and frame.moment - self.pending[0].moment >= .5):
            oldest = self.pending.popleft()
            ready.append(self._render(oldest, frame.moment - oldest.moment < .5,
                                      'capacity' if len(self.pending) >= 16 else 'media-window'))
        if self.redactor.debug:
            self.redactor.detection_stats['lookahead_max_frames'] = max(
                self.redactor.detection_stats['lookahead_max_frames'], len(self.pending))
        return ready

    def wait_seconds(self):
        if self.pending and self.pending[0].deadline is not None:
            return max(0, self.pending[0].deadline - time.monotonic())
        return None

    def expire(self):
        ready, now = [], time.monotonic()
        while self.pending and self.pending[0].deadline is not None and self.pending[0].deadline <= now:
            ready.append(self._render(self.pending.popleft(), True, 'wall-deadline'))
        return ready

    def flush(self, *, reason='flush'):
        result = []
        while self.pending:
            oldest = self.pending.popleft()
            result.append(self._render(oldest, True, reason))
        return result

    def _render(self, frame, incomplete, reason):
        if self.trace:
            self.trace.update_frame(frame.trace_id, lookahead_released_monotonic=time.monotonic(),
                release_reason=reason, incomplete_future_window=incomplete,
                lookahead_depth_at_release=len(self.pending))
        # EOF, scene cuts and burst caps cannot rely on a complete future window.
        if incomplete and frame.discover is not None:
            with self.trace.span('op:lookahead-discover') if self.trace else nullcontext():
                frame.extra.extend(frame.discover())
            if self.redactor.debug:
                self.redactor.detection_stats['lookahead_tail_checks'] += 1
        return frame.render(frame.extra)
