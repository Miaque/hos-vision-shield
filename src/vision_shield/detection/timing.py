"""Bounded host-wall counters; nested/concurrent scopes must not be added together."""

from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import Lock
from time import perf_counter


STAGES = (
    "prepare-primary", "prepare-detail", "forward-primary-face", "forward-primary-plate",
    "forward-detail-face", "forward-detail-plate",
    "post-primary", "post-detail", "local-confirm", "input-copy", "output-copy", "explicit-sync",
    "track-confirm", "track-update", "track-flow", "track-probe", "track-recheck", "track-face-recheck",
    "color-convert", "gray-copy", "mask-gpu",
    "source-read", "source-decode", "source-backpressure", "session-primary-face", "session-detail-face",
    "plate-work-check",
    "wait-primary",
)


class StageTimes:
    def __init__(self):
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.lock = Lock()
        self.observer = None
        self.span_observer = None
        self.primary_observer = None

    @contextmanager
    def measure(self, name):
        with self.span_observer('stage:' + name) if self.span_observer else nullcontext():
            started = perf_counter()
            try:
                yield
            finally:
                elapsed = perf_counter() - started
                with self.lock:
                    self.seconds[name] += elapsed
                if self.observer is not None:
                    self.observer(name, elapsed)

    def snapshot(self):
        with self.lock:
            return self.seconds.copy()

    def reset(self):
        with self.lock:
            self.seconds = dict.fromkeys(STAGES, 0.0)


def measure(owner, name):
    timing = getattr(owner, "stage_times", None)
    return timing.measure(name) if timing is not None else nullcontext()


def timed(name):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with measure(self, name):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


def snapshot(owner):
    timing = getattr(owner, "stage_times", None)
    return timing.snapshot() if timing is not None else {}
