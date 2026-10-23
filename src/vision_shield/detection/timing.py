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
)


class StageTimes:
    def __init__(self):
        self.seconds = dict.fromkeys(STAGES, 0.0)
        self.lock = Lock()

    @contextmanager
    def measure(self, name):
        started = perf_counter()
        try:
            yield
        finally:
            with self.lock:
                self.seconds[name] += perf_counter() - started

    def snapshot(self):
        with self.lock:
            return self.seconds.copy()


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
