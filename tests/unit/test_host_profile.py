"""The diagnostic harness must count real calls without changing detector behavior."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


from vision_shield.detection import host_profile as profile
from vision_shield.detection import Detection


def test_ort_nodes_match_the_sampled_real_call_and_bound_the_report():
    samples = [dict(call=2)]
    events = [dict(cat='Session', name='model_run', ts=0, dur=100),
              dict(cat='Session', name='model_run', ts=1000, dur=2000),
              dict(cat='Node', name='outside_kernel_time', ts=10, dur=90,
                   args=dict(op_name='Conv', provider='CUDAExecutionProvider'))]
    events += [dict(cat='Node', name=f'conv{i}_kernel_time', ts=1000 + i * 100, dur=80 + i,
                    args=dict(op_name='Conv', provider='CUDAExecutionProvider')) for i in range(15)]
    profile.attach_ort_nodes(samples, events)
    assert samples[0]['ort_model_run_ms'] == 2
    nodes = samples[0]['ort_top_nodes']
    assert len(nodes) == 8 and nodes[0]['name'] == 'conv14_kernel_time'
    assert nodes[0]['host_ms'] == .094 and nodes[0]['provider'] == 'CUDAExecutionProvider'
    assert all(row['name'] != 'outside_kernel_time' for row in nodes)

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


def test_statistics_expose_tail_not_hidden_by_average():
    summary = profile.statistics([1.0] * 99 + [1600.0])
    assert summary['p99_ms'] == pytest.approx(16.99)
    assert summary['max_ms'] == 1600
    assert summary['over_500_ms'] == summary['over_1000_ms'] == 1
    assert profile.statistics([])['max_ms'] is None


def test_local_face_observer_keeps_roi_shape_session_and_phase_identity(tmp_path, monkeypatch):
    from contextlib import ExitStack
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch

    sentinel = [np.ones((1, 1), np.float32)]
    ended = []

    class Session:
        def get_providers(self):
            return ['CUDAExecutionProvider']

        def run(self, outputs, feed):
            assert feed['input'].shape == (1, 3, 32, 64)
            return sentinel

        def end_profiling(self):
            ended.append(True)
            target = tmp_path / 'ort.json'
            target.write_text('[]', 'utf-8')
            return str(target)

    monkeypatch.setattr(ort, 'InferenceSession', Session)
    session = Session()
    collector = profile.ProfileTimes(tmp_path)
    collector.ort_session = session
    monkeypatch.setattr(profile, 'MAX_ORT_CALLS', 1, raising=False)
    with ExitStack() as stack:
        profile.instrument_local_faces(stack, collector)
        with profile.scope(frame=17, purpose='local-confirm', local_id='17:1',
                           local_face=True, local_details={}):
            pixels = torch.zeros((14, 29, 3), dtype=torch.uint8).cpu().numpy()
            pixels = cv2.resize(pixels, (64, 32))
            blob = cv2.dnn.blobFromImage(pixels)
            assert session.run(None, {'input': blob}) is sentinel
            assert session.run(None, {'input': blob}) is sentinel
    assert ended == [True] and collector.ort_session is None
    collector.close()
    rows = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert {'local-readback', 'local-numpy', 'local-resize', 'ort-local-face'} <= {r['stage'] for r in rows}
    runs = [r for r in rows if r['stage'] == 'ort-local-face']
    assert [r['first_local_shape'] for r in runs] == [True, False]
    assert runs[0]['frame'] == 17 and runs[0]['local_id'] == '17:1'
    assert runs[0]['roi_shape'] == [14, 29, 3]
    assert runs[0]['resized_shape'] == [32, 64, 3]
    assert runs[0]['input_shape'] == [1, 3, 32, 64]
    assert runs[0]['provider'] == 'CUDAExecutionProvider'
    assert runs[0]['session_instance'] == runs[1]['session_instance']
    assert collector.report(1)['local_faces']['timing']['calls'] == 2
    assert len(collector.report(1)['local_faces']['longest']) == 2


def test_local_trace_discards_fast_call_and_keeps_slow_call_in_fixed_slot(tmp_path, monkeypatch):
    import torch.profiler

    stage = 'ort-local-face'
    durations = iter([2000, 1600000])

    @contextmanager
    def profiler(**kwargs):
        duration = next(durations)
        trace = {'traceEvents': [
            dict(ph='X', cat='user_annotation', name=stage, tid=1, ts=0, dur=duration),
            dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', tid=1, ts=1, dur=2, args={'correlation': 7}),
            dict(ph='X', cat='kernel', name='actual-local-kernel', tid=2, ts=10, dur=5, args={'correlation': 7}),
        ]}
        yield SimpleNamespace(events=lambda: [SimpleNamespace(name=stage, cpu_time_total=duration)],
                              export_chrome_trace=lambda target: Path(target).write_text(json.dumps(trace)))

    monkeypatch.setattr(torch.profiler, 'profile', profiler)
    collector = profile.ProfileTimes(tmp_path, device_stage='local-face')
    with profile.scope(frame=40, purpose='local-confirm'), collector.trace(stage, 3, slow_only=True):
        pass
    assert not list(tmp_path.glob('*.json'))
    with profile.scope(frame=90, purpose='local-confirm'), collector.trace(stage, 9, slow_only=True):
        pass
    assert (tmp_path / 'local-face-64.json').is_file()
    assert collector.device_samples[0]['call'] == 9
    assert collector.device_samples[0]['host_forward_ms'] == 1600
    collector.close()


def test_nested_confirmation_keeps_parent_for_readback_and_run_correlation(tmp_path, monkeypatch):
    from contextlib import ExitStack
    import numpy as np
    from vision_shield.detection.quality import Detector
    from vision_shield.redaction import FrameRedactor

    monkeypatch.setattr(Detector, '_confirmed', lambda *args, **kwargs: True)

    def recheck(instance, pixels, box, *args):
        return Detector._confirmed(object(), pixels, Detection('face', box, 0.5), object())

    monkeypatch.setattr(FrameRedactor, '_recheck', recheck)
    collector = profile.ProfileTimes(tmp_path)
    try:
        with ExitStack() as stack:
            profile.instrument(stack, collector)
            with profile.scope(frame=17):
                assert FrameRedactor._recheck(object(), np.zeros((32, 32, 3), np.uint8), (1, 2, 8, 9))
    finally:
        collector.close()
    rows = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    child, parent = rows
    assert parent['parent_local_id'] is None
    assert child['parent_local_id'] == parent['local_id']
    assert child['local_id'] != parent['local_id']
    assert child['target_box'] == parent['target_box'] == [1, 2, 8, 9]


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
