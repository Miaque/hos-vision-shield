"""模型计时留在内部事件，失败不产生成功媒体。"""

import io
import json
from types import SimpleNamespace

import pytest

from vision_shield.config import Settings
from vision_shield.media import child
from vision_shield.jobs.tasks import log_model_timing
from vision_shield.api.schemas import Stats


@pytest.mark.parametrize("load_fails", [False, True])
def test_child_model_timing_and_failure_have_no_success(tmp_path, monkeypatch, capsys, load_fails):
    from vision_shield import detection
    payload = dict(kind="image", source="rtsp://private:secret@camera/input", directory=str(tmp_path / "attempt"),
                   config=Settings(_env_file=None).model_dump(mode="json"))
    monkeypatch.setattr(child.sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
    monkeypatch.setattr(child.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None))
    def load(config):
        if load_fails:
            raise RuntimeError("private:secret")
        return object()
    monkeypatch.setattr(detection, "load_detector", load)
    with pytest.raises(SystemExit):
        child.main()
    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    timings = [event for event in events if "model_load_seconds" in event]
    assert bool(timings) is not load_fails
    if timings:
        assert timings[0]["model_load_seconds"] >= 0
    assert not any("result_ref" in event for event in events)
    assert "secret" not in output


def test_model_load_time_is_logged_without_changing_response_stats(caplog):
    import logging
    event = dict(model_load_seconds=1.25, source="rtsp://private:secret@camera")
    with caplog.at_level(logging.INFO):
        log_model_timing(event, "job-id", 3)
    assert "job-id" in caplog.text and "1.250" in caplog.text and "secret" not in caplog.text
    assert set(Stats.model_validate(event).model_dump()) == {"processed_frames", "dropped_frames", "elapsed_seconds"}


async def test_task_stage_timings_are_headers_and_do_not_change_the_json_contract():
    from unittest.mock import AsyncMock
    import httpx
    from vision_shield.main import create_app

    record = dict(id="00000000-0000-0000-0000-000000000001", kind="video", state="processing",
                  desired_state="run", ready=False, created_at="2026-10-22T00:00:00Z",
                  updated_at="2026-10-22T00:00:01Z", error=None,
                  stats=dict(processed_frames=1, dropped_frames=0, elapsed_seconds=2),
                  timings=dict(frame=1.5, inference=1.25, encode=0.1), source_ref="rtsp://private:secret@camera")
    app = create_app(Settings(_env_file=None))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        app.state.store.get = AsyncMock(return_value=record)
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert response.status_code == 200
        assert response.headers["server-timing"] == "task-frame;dur=1500.000, task-inference;dur=1250.000, task-encode;dur=100.000"
        assert "timings" not in response.json() and response.json()["stats"] == record["stats"]
        assert "secret" not in response.text and "secret" not in str(response.headers)
        record["timings"].update({"candidate-primary": 0.8, "candidate-detail": 0.6, "private-secret": 123})
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert response.headers["server-timing"].endswith("task-candidate-primary;dur=800.000, task-candidate-detail;dur=600.000")
        assert "secret" not in str(response.headers) and "timings" not in response.json()
        from vision_shield.detection.timing import STAGES
        record["timings"].update(dict.fromkeys(STAGES, 0.01))
        response = await client.get(f"/v1/tasks/{record['id']}")
        for name in STAGES:
            assert f"task-{name};dur=10.000" in response.headers["server-timing"]
        assert "secret" not in str(response.headers) and "timings" not in response.json()
        record["video_encoder"] = "h264_nvenc"
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert response.headers["x-video-encoder"] == "h264_nvenc"
        assert "video_encoder" not in response.json()
        record["video_decoder"] = "h264_nvdec"
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert response.headers["x-video-decoder"] == "h264_nvdec" and "video_decoder" not in response.json()
        record["video_decoder"] = "private-secret"
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert "x-video-decoder" not in response.headers
        record["video_encoder"] = "private-secret"
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert "x-video-encoder" not in response.headers
        del record["timings"]
        response = await client.get(f"/v1/tasks/{record['id']}")
        assert "server-timing" not in response.headers


def test_child_attaches_requested_sampler_and_finishes_only_after_video(tmp_path, monkeypatch, capsys):
    from vision_shield import detection
    from vision_shield.media import video

    primary = SimpleNamespace(device=SimpleNamespace(type='cuda'))
    detector = SimpleNamespace(primary=primary)
    source = tmp_path / 'source.mp4'
    payload = dict(kind='video', source=str(source), directory=str(tmp_path / 'attempt'),
                   profile_detection=True, config=Settings(_env_file=None).model_dump(mode='json'))
    monkeypatch.setattr(child.sys, 'stdin', io.StringIO(json.dumps(payload) + '\n'))
    monkeypatch.setattr(child.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(detection, 'load_detector', lambda cfg: detector)
    def process(source, target, actual, config, stop, emit):
        from pathlib import Path
        assert actual is detector and primary.device_samples.samples == []
        assert not any('device_profile' in json.loads(line) for line in capsys.readouterr().out.splitlines())
        Path(target).write_bytes(b'video')
    monkeypatch.setattr(video, 'process_video', process)
    child.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]['device_profile'] == dict(state='partial', samples=[])
    assert events[1]['result_ref'].endswith('result.mp4')


def test_inference_timing_includes_local_rechecks(monkeypatch):
    import numpy as np
    from vision_shield import redaction

    clock = [0.0]
    monkeypatch.setattr(redaction.time, "monotonic", lambda: clock[0])
    class Once:
        calls = 0
        candidate_timings = {"candidate-primary": 0.03, "candidate-detail": 0.01}

        def detect(self, pixels):
            self.calls += 1
            clock[0] += 0.05
            return [(20, 20, 40, 40)] if self.calls == 1 else []
    detector = Once()
    redactor = redaction.FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    redactor.apply(pixels, 0.05)
    assert detector.calls == 3, "两次全图和一次局部确认都应实际运行"
    assert redactor.inference_seconds == pytest.approx(0.15)
    assert redactor.candidate_timings == pytest.approx({"candidate-primary": 0.09, "candidate-detail": 0.03})


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('mode', ['host', 'forward-detail-face'])
def test_child_host_profile_is_opt_in_and_failure_does_not_publish_output(tmp_path, monkeypatch, capsys, fails, mode):
    from pathlib import Path
    from vision_shield import detection
    from vision_shield.media import video
    # Import numerical libraries before replacing the private control-pipe thread.
    from vision_shield.detection import host_profile, egoblur, cuda_detail, quality

    primary = SimpleNamespace(candidates=lambda pixels, **kw: [])
    detail = SimpleNamespace(candidates=lambda pixels, **kw: [])
    detector = SimpleNamespace(primary=primary, detail=detail)
    payload = dict(kind='video', source='private-source', directory=str(tmp_path / 'attempt'),
                   profile_detection=mode, config=Settings(_env_file=None, inference_device='cuda', detection_backend='quality_yolo').model_dump(mode='json'))
    monkeypatch.setattr(child.sys, 'stdin', io.StringIO(json.dumps(payload) + '\n'))
    monkeypatch.setattr(child.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(detection, 'load_detector', lambda cfg: detector)
    monkeypatch.setattr(host_profile, 'runtime_metadata', lambda *args: dict(backend='quality'))
    original = video.av.open

    def process(source, target, actual, config, stop, emit):
        assert actual is detector and video.av.open is not original
        assert not hasattr(primary, 'device_samples')
        if fails:
            raise RuntimeError('private-source')
        Path(target).write_bytes(b'processed-video')
    monkeypatch.setattr(video, 'process_video', process)
    if fails:
        with pytest.raises(SystemExit):
            child.main()
    else:
        child.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    report, = [event['device_profile'] for event in events if 'device_profile' in event]
    assert report['state'] == ('failed' if fails else 'complete' if mode == 'host' else 'partial')
    assert report['mode'] == mode
    assert any('result_ref' in event for event in events) is not fails
    assert (tmp_path / 'attempt/result.mp4').exists() is not fails
    assert video.av.open is original
