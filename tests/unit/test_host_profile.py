"""The diagnostic harness must count real calls without changing detector behavior."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


from vision_shield.detection import host_profile as profile
from vision_shield.detection import Detection

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
            return [Detection(kind, (1, 2, 3, 4), 0.5)]

    model = Model()
    profile.instrument_candidates(model, "detail", collector)
    with profile.scope(frame=1, purpose="discovery"):
        assert model.candidates(SimpleNamespace(shape=(1080, 3840, 3)), kind="face")[0].kind == "face"
    with profile.scope(frame=2, purpose="local-confirm", local_calls={}):
        for _ in range(2):
            assert model.candidates(SimpleNamespace(shape=(320, 320, 3)), kind="face")[0].kind == "face"
    collector.close()
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert [row["view"] for row in rows[:2]] == ["global", "tile"]
    assert [row["mirror"] for row in rows[2:]] == [False, False, True, True]
    assert collector.report(2)["views"]["discovery"]["tile"] == 1
    evidence = collector.report(2)["candidate_evidence"]["rows"]
    assert len(evidence) == 1 and evidence[0]["stage"] == "detail"
    assert evidence[0]["detections"][0]["score"] == 0.5


def test_statistics_keep_first_and_warmup_separate():
    summary = profile.statistics(list(range(1, 21)))
    assert summary["calls"] == 20
    assert summary["first_ms"] == 1
    assert summary["p50_ms"] == 10.5
    assert summary["p95_ms"] == pytest.approx(19.05)
    assert summary["steady_after_10_calls"]["calls"] == 10
    assert summary["steady_after_10_calls"]["p50_ms"] == 15.5


def test_candidate_evidence_is_bounded_and_keeps_empty_discovery_results(tmp_path):
    collector = profile.ProfileTimes(tmp_path)
    detections = [Detection("face", (1, 2, 30, 40), 0.25)]
    with profile.scope(frame=1, source_seconds=51.833, purpose="discovery"):
        collector.record_candidates("primary", detections)
        collector.record_candidates("detail", [])
    with profile.scope(frame=1, purpose="local-confirm"):
        collector.record_candidates("primary", detections)
    with profile.scope(frame=33, purpose="discovery"):
        collector.record_candidates("primary", detections)
    evidence = collector.report(33)["candidate_evidence"]
    assert len(evidence["rows"]) == 2
    assert evidence["rows"][0]["detections"] == [{"kind": "face", "box": [1, 2, 30, 40], "score": 0.25}]
    assert evidence["rows"][0]["source_seconds"] == 51.833
    assert evidence["rows"][1]["detections"] == []
    with profile.scope(frame=2, purpose="discovery"):
        collector.record_candidates("primary", detections * 2000)
    assert collector.report(33)["candidate_evidence"]["truncated"]
    assert len(collector.report(33)["candidate_evidence"]["rows"]) == 2
    collector.close()


def test_final_mask_evidence_keeps_pixel_bounds_empty_frames_and_budget(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from vision_shield.redaction import FrameRedactor

    collector = profile.ProfileTimes(tmp_path)
    boxes = [(-0.2, 1.8, 12.3, 25.1), (22, 1, 23, 4)]
    monkeypatch.setattr(FrameRedactor, "regions", lambda self, *args, **kwargs: boxes)
    with ExitStack() as stack:
        profile.instrument(stack, collector)
        pixels = SimpleNamespace(shape=(20, 16, 3))
        assert FrameRedactor.regions(object(), pixels, 0.0) is boxes
        boxes.clear()
        assert FrameRedactor.regions(object(), pixels, 0.1) is boxes
    evidence = collector.report(2)["mask_evidence"]
    assert evidence["rows"] == [[1, [[0, 1, 13, 20]]], [2, []]]
    assert evidence["shape"] == [20, 16] and not evidence["truncated"]
    collector.mask_bytes = evidence["byte_limit"]
    with profile.scope(frame=3):
        collector.record_masks(pixels.shape, [])
    evidence = collector.report(3)["mask_evidence"]
    assert evidence["truncated"] and len(evidence["rows"]) == 2
    collector.close()


def test_batched_view_forward_is_not_attributed_to_the_last_tile(tmp_path):
    from itertools import batched

    collector = profile.ProfileTimes(tmp_path)
    bounds = [(0, 0, 3840, 1080), (0, 0, 1920, 1080),
              (960, 0, 2880, 1080), (1920, 0, 3840, 1080)]

    class Model:
        parallel, tile_side, _plate_session = True, 1920, object()

        def _views(self, *args):
            yield from bounds

        def candidates(self, pixels, **kwargs):
            for group in batched(self._views(*pixels.shape[:2][::-1], self.tile_side), 4):
                assert list(group) == bounds
                with collector.measure("forward-detail-plate"):
                    pass
            return []

    model = Model()
    profile.instrument_candidates(model, "detail", collector)
    with profile.scope(frame=17, purpose="discovery"):
        assert model.candidates(SimpleNamespace(shape=(1080, 3840, 3), is_cuda=True)) == []
    collector.close()
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["frame"] == 17
    assert rows[0]["view"] == "mixed" and rows[0]["source_region"] is None
    assert rows[0]["batched_candidate_regions"] == [list(b) for b in bounds]
    assert collector.report(1)["views"]["discovery"] == {"global": 1, "tile": 3}


def test_diagnostic_budget_stops_before_running_more_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "MAX_MEASUREMENTS", 1)
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
