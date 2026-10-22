from concurrent.futures import ThreadPoolExecutor
from threading import local
from types import SimpleNamespace

import pytest

from vision_shield.detection.timing import STAGES, StageTimes, measure, snapshot, timed


def test_wall_counters_cover_concurrent_calls_and_failures(monkeypatch):
    from vision_shield.detection import timing

    clock = iter(range(6))
    monkeypatch.setattr(timing, "perf_counter", lambda: next(clock))
    counter = StageTimes()
    owner = SimpleNamespace(stage_times=counter)
    with measure(owner, "forward-primary-plate"):
        pass
    with pytest.raises(ValueError):
        with measure(owner, "local-confirm"):
            raise ValueError("expected")
    with ThreadPoolExecutor(1) as pool:
        def work():
            with measure(owner, "forward-primary-plate"):
                pass
        pool.submit(work).result()
    snapshot = counter.snapshot()
    assert set(snapshot) == set(STAGES)
    assert snapshot["forward-primary-plate"] == 2 and snapshot["local-confirm"] == 1
    snapshot["forward-primary-plate"] = 100
    assert counter.snapshot()["forward-primary-plate"] == 2
    with measure(object(), "forward-primary-plate"):
        pass


def test_nested_method_scopes_keep_return_values_and_do_not_claim_exclusive_time(monkeypatch):
    from vision_shield.detection import timing

    clock = iter([0, 1, 2, 4])
    monkeypatch.setattr(timing, "perf_counter", lambda: next(clock))

    class Detector:
        stage_times = StageTimes()

        @timed("local-confirm")
        def detect(self):
            with measure(self, "post-primary"):
                return [1, 2, 3]

    detector = Detector()
    assert detector.detect() == [1, 2, 3]
    assert snapshot(detector)["local-confirm"] == 4
    assert snapshot(detector)["post-primary"] == 1
    assert snapshot(object()) == {}


def test_concurrent_model_threads_keep_all_elapsed_samples(monkeypatch):
    from vision_shield.detection import timing

    clock = local()
    def tick():
        clock.value = getattr(clock, "value", 0) + 1
        return clock.value
    monkeypatch.setattr(timing, "perf_counter", tick)
    counter = StageTimes()
    def work(_):
        with counter.measure("prepare-detail"):
            pass
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(200)))
    assert counter.snapshot()["prepare-detail"] == 200
