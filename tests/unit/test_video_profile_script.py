"""The diagnostic harness must count real calls without changing detector behavior."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[2] / "scripts/profile-video.py"
spec = importlib.util.spec_from_file_location("profile_video", path)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_parallel_scopes_keep_frame_view_and_actual_forward_counts(tmp_path):
    collector = profile.ProfileTimes(tmp_path, clock=iter(range(20)).__next__)
    with profile.scope(frame=7, purpose="discovery", view="tile"):
        with profile.ContextPool(1) as pool:
            def run():
                with collector.measure("forward-detail-face"):
                    return "unchanged"
            assert pool.submit(run).result() == "unchanged"
    with pytest.raises(ValueError):
        with profile.scope(frame=8, purpose="local-confirm", view="global"):
            with collector.measure("forward-primary-plate"):
                raise ValueError("failed forward")
    collector.close()
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [(r["stage"], r["frame"], r["purpose"], r["view"], r["failed"]) for r in rows] == [
        ("forward-detail-face", 7, "discovery", "tile", False),
        ("forward-primary-plate", 8, "local-confirm", "global", True),
    ]
    assert rows[0]["host_ms"] == 1000
    stats = collector.report(2)
    assert stats["models"]["forward-detail-face"]["calls"] == 1
    assert stats["models"]["forward-primary-plate"]["failures"] == 1
    assert stats["models"]["forward-primary-plate"]["calls_per_frame"] == 0.5
    assert collector.snapshot()["forward-primary-plate"] == 1
    with collector.measure("candidate-detail"):
        pass
    assert "candidate-detail" not in collector.snapshot()


def test_views_and_local_mirror_are_distinguished_without_extra_calls(tmp_path):
    collector = profile.ProfileTimes(tmp_path)

    class Model:
        @staticmethod
        def _views(width, height, side=1920):
            yield 0, 0, width, height
            yield 0, 0, side, height

        def candidates(self, pixels, *, kind=None):
            for _ in self._views(*pixels.shape[:2][::-1]):
                with profile.ContextPool(1) as pool:
                    def run():
                        with collector.measure("forward-detail-face"):
                            pass
                    pool.submit(run).result()
            return [kind]

    model = Model()
    profile.instrument_candidates(model, "detail", collector)
    with profile.scope(frame=1, purpose="discovery"):
        assert model.candidates(SimpleNamespace(shape=(1080, 3840, 3)), kind="face") == ["face"]
    with profile.scope(frame=2, purpose="local-confirm", local_calls={}):
        for _ in range(2):
            assert model.candidates(SimpleNamespace(shape=(320, 320, 3)), kind="face") == ["face"]
    collector.close()
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert [row["view"] for row in rows[:2]] == ["global", "tile"]
    assert [row["mirror"] for row in rows[2:]] == [False, False, True, True]
    assert collector.report(2)["views"]["discovery"]["tile"] == 1


def test_statistics_keep_first_and_warmup_separate():
    summary = profile.statistics(list(range(1, 21)))
    assert summary["calls"] == 20
    assert summary["first_ms"] == 1
    assert summary["p50_ms"] == 10.5
    assert summary["p95_ms"] == pytest.approx(19.05)
    assert summary["steady_after_10_calls"]["calls"] == 10
    assert summary["steady_after_10_calls"]["p50_ms"] == 15.5


def test_diagnostic_budget_stops_before_running_more_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(profile.observer, "MAX_MEASUREMENTS", 1)
    collector = profile.ProfileTimes(tmp_path)
    with collector.measure("forward-primary-face"):
        pass
    executed = False
    with pytest.raises(RuntimeError, match="budget"):
        with collector.measure("forward-primary-face"):
            executed = True
    assert not executed
    assert collector.report(1)["models"]["forward-primary-face"]["calls"] == 1
    collector.close()


def test_codec_wrappers_keep_packets_and_stream_configuration(tmp_path):
    collector = profile.ProfileTimes(tmp_path)
    frames, packets = [object()], [object()]
    audio = SimpleNamespace(stream=SimpleNamespace(type="audio"))
    video = SimpleNamespace(stream=SimpleNamespace(type="video"), decode=lambda: frames)
    stream = SimpleNamespace(encode=lambda *args: packets)
    container = SimpleNamespace(demux=lambda: iter([audio, video]), add_stream=lambda: stream, mux=lambda p: p)
    wrapped = profile.TimedContainer(container, collector)
    untouched_audio, timed_video = list(wrapped.demux())
    assert untouched_audio is audio
    assert timed_video.decode() is frames
    encoder = wrapped.add_stream()
    encoder.width = 1920
    assert stream.width == encoder.width == 1920
    assert encoder.encode(frames[0]) is packets
    assert encoder.encode() is packets
    assert wrapped.mux(audio) is audio
    assert collector.report(0)["host_stages"]["decode-host"]["calls"] == 1
    assert collector.report(0)["host_stages"]["encode-host"]["calls"] == 2
    collector.close()


def test_device_trace_uses_actual_model_name_and_restores_shared_helper(tmp_path, monkeypatch):
    import torch.profiler
    from vision_shield.detection import device_profile

    stage, names = "forward-detail-face", []
    trace = {"traceEvents": [
        dict(ph="X", cat="user_annotation", name=stage, tid=1, ts=0, dur=100),
        dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", tid=1, ts=1, dur=2, args={"correlation": 7}),
        dict(ph="X", cat="kernel", name="actual-kernel", tid=2, ts=10, dur=5, args={"correlation": 7}),
    ]}

    @contextmanager
    def profiler(**kwargs):
        yield SimpleNamespace(export_chrome_trace=lambda target: Path(target).write_text(json.dumps(trace)))

    @contextmanager
    def record(name):
        names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "profile", profiler)
    monkeypatch.setattr(torch.profiler, "record_function", record)
    original = device_profile.SCOPE
    collector = profile.ProfileTimes(tmp_path)
    with profile.scope(frame=17, purpose="local-confirm", source_shape=[320, 320, 3]), collector.trace(stage, 2):
        pass
    assert names == [stage]
    assert collector.device_samples[0]["stage"] == stage
    assert collector.device_samples[0]["kernel_count"] == 1
    assert collector.device_samples[0]["context"]["frame"] == 17
    assert collector.device_samples[0]["context"]["source_shape"] == [320, 320, 3]
    assert device_profile.SCOPE == original
    collector.close()
