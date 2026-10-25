import httpx
import pytest
from redis.asyncio import Redis

from vision_shield.config import Settings
from vision_shield.main import create_app
from vision_shield.jobs import tasks


@pytest.mark.parametrize("input_scheme", ["rtsp", "rtmp"])
@pytest.mark.parametrize("publish_url", [None, "rtsp://127.0.0.1/custom?token=secret", "rtmp://127.0.0.1/live/custom?sign=secret"])
@pytest.mark.parametrize("cadence", [False, True])
@pytest.mark.parametrize("playback_url", [None, 'https://play.example/live/camera.m3u8?token=play'])
async def test_stream_intent_is_stoppable_without_a_file_result(
    redis_url, tmp_path, publish_url, cadence, playback_url, input_scheme,
):
    app = create_app(Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        redis = app.state.redis
        await redis.flushdb()
        seconds, micros = await redis.time()
        await redis.zadd("vs:workers:streams", {"test": seconds * 1000 + micros // 1000})
        body = {"url": f"{input_scheme}://127.0.0.1/source"}
        if publish_url:
            body["publish_url"] = publish_url
        if playback_url:
            body['playback_url'] = playback_url
        headers = {"Idempotency-Key": "stream"}
        if cadence:
            headers.update({"X-Detection-Interval": "3", "X-Tracking-Mode": "flow"})
        created = await client.post("/v1/stream-tasks", json=body, headers=headers)
        assert created.status_code == 202
        task = created.json()
        assert task['playback_url'] == (playback_url or publish_url or f"rtsp://127.0.0.1:8554/redacted/{task['id']}")
        assert not task['ready'] and 'publish_url' not in task
        stored = await app.state.store.get(task["id"])
        assert stored.get("publish_url") == publish_url
        assert stored.get('playback_url') == playback_url
        assert "max_bitrate" not in stored
        assert stored.get("detection_interval", 1) == (3 if cadence else 1)
        assert stored.get("tracking", "template") == ("flow" if cadence else "template")
        assert (await client.post("/v1/stream-tasks", json=body, headers=headers)).json()["id"] == task["id"]
        if not cadence:
            defaults = headers | {"X-Detection-Interval": "1", "X-Tracking-Mode": "template"}
            assert (await client.post("/v1/stream-tasks", json=body, headers=defaults)).json()["id"] == task["id"]
        other_headers = {"Idempotency-Key": "stream"} if cadence else headers | {"X-Detection-Interval": "3", "X-Tracking-Mode": "flow"}
        assert (await client.post("/v1/stream-tasks", json=body, headers=other_headers)).status_code == 409
        changed = body | {"publish_url": "rtmp://127.0.0.1/live/other"}
        assert (await client.post("/v1/stream-tasks", json=changed, headers=headers)).status_code == 409
        changed = body | {'playback_url': 'https://play.example/other.m3u8'}
        assert (await client.post('/v1/stream-tasks', json=changed, headers=headers)).status_code == 409
        queried = await client.post('/v1/tasks/query', json={'task_ids': [task['id']]})
        assert queried.json()['items'][0]['playback_url'] == task['playback_url']
        assert (await client.get(f"/v1/tasks/{task['id']}/result")).status_code == 409
        for _ in range(2):
            stopped = await client.patch(f"/v1/tasks/{task['id']}", json={"desired_state": "stopped"})
            assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"


async def test_stream_max_bitrate_is_stored_when_explicit(redis_url, tmp_path):
    app = create_app(Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        seconds, micros = await app.state.redis.time()
        await app.state.redis.zadd("vs:workers:streams", {"test": seconds * 1000 + micros // 1000})
        response = await client.post('/v1/stream-tasks', json={
            'url': 'rtsp://127.0.0.1/source', 'max_bitrate': 20_000_000,
        }, headers={'Idempotency-Key': 'stream-bitrate'})
        assert response.status_code == 202
        stored = await app.state.store.get(response.json()['id'])
        assert stored['max_bitrate'] == 20_000_000


@pytest.mark.parametrize("scheme", ["rtsp", "rtmp"])
@pytest.mark.parametrize("cadence", [False, True])
async def test_direct_publisher_stop_uses_process_exit_confirmation(redis_url, tmp_path, monkeypatch, scheme, cadence):
    import asyncio
    import time

    settings = Settings(_env_file=None, debug=True, redis_url=redis_url, media_root=tmp_path,
                        source_allowlist=["127.0.0.1"], mediamtx_control_url="http://127.0.0.1:1")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await app.state.redis.flushdb()
        store = app.state.store
        target = f"{scheme}://127.0.0.1/app/camera?sign=secret"
        options = dict(detection_interval=3, tracking="flow", max_bitrate=20_000_000) if cadence else {"max_bitrate": 20_000_000}
        task = await store.create("stream", "rtsp", "rtsp://127.0.0.1/source", publish_url=target, **options)
        owned = await store.claim(task["id"], "owner")
        cancel = asyncio.Event()

        async def run(command, payload, cancel, on_event, timeout, *, on_exit=None):
            assert payload["publish_url"] == target and payload["config"]["publish_url"] == target
            assert payload["config"]["max_bitrate"] == 20_000_000
            assert payload["config"].get("detection_interval", 1) == (3 if cadence else 1)
            assert payload["config"].get("tracking", "template") == ("flow" if cadence else "template")
            await on_event(dict(output=True, output_monotonic=time.monotonic(), ready=True))
            from fractions import Fraction
            from types import SimpleNamespace
            from vision_shield.media.stream_metrics import StreamMetrics

            metrics = StreamMetrics(debug=True)
            empty_stats = dict(input_frames=0, queue_frames=0, audio_queue_bytes=0,
                               failed_frames=0, protected_frames=0, **metrics.snapshot())
            await on_event(dict(stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0,
                                          stream=empty_stats)))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.get('/v1/tasks/' + task['id'])
                assert response.status_code == 200
                assert response.json()['stats']['stream']['recent_outputs'] == []
            metrics.submit(SimpleNamespace(pts=0, time_base=Fraction(1, 1000)), 10.0, Fraction(9))
            metrics.written(Fraction(0), 10.25, encoded_bytes=100, encoded_vcl_sha256='a' * 64)
            metrics.submit(SimpleNamespace(pts=40, time_base=Fraction(1, 1000)), 10.04, Fraction(9),
                           timeline_pts=Fraction(226, 25), timestamp_repaired=True)
            metrics.written(Fraction(40, 1000), 10.29)
            stream_stats = dict(input_frames=3, queue_frames=0, audio_queue_bytes=0,
                                failed_frames=0, protected_frames=0, **metrics.snapshot())
            await on_event(dict(stats=dict(processed_frames=3, dropped_frames=0, elapsed_seconds=1, stream=stream_stats),
                                video_decoder="h264_nvdec", video_encoder="h264_nvenc",
                                detection_stats=dict(full_frames=1, skipped_frames=2)))
            current = await store.get(task["id"])
            assert current["ready"] and current["state"] == "running"
            assert current["video_decoder"] == "h264_nvdec" and current["video_encoder"] == "h264_nvenc"
            assert current["detection_stats"] == dict(full_frames=1, skipped_frames=2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.get('/v1/tasks/' + task['id'])
                assert response.status_code == 200
                assert response.json()['stats']['stream'] == stream_stats
            await store.request_stop(task["id"], "stopped")
            await on_event(dict(failure_type="BufferError", failure_at=["process_stream:200"],
                                error={"message": "rtsp://private/?token=secret"}))
            failure = (await store.get(task["id"]))["media_failure"]
            assert failure["type"] == "BufferError" and failure["at"] == ["process_stream:200"]
            assert failure["generation"] == owned["generation"]
            assert "secret" not in str(failure)
            cancel.set()
            if on_exit:
                await on_exit()
            return dict(cancelled=True, returncode=0)

        monkeypatch.setattr(tasks, "run_process", run)
        await tasks.run_stream(owned, store, settings, cancel, "owner", owned["generation"])
        current = await store.get(task["id"])
        assert current["state"] == "stopped" and not current["publish_allowed"]
        assert current["publish_url"] == target and not current["sessions"]


async def test_stream_failure_diagnostics_are_available_without_exposing_source(tmp_path):
    from unittest.mock import AsyncMock

    app = create_app(Settings(_env_file=None, media_root=tmp_path, debug=True))
    failure = dict(type="BufferError", at=["process_stream:200"], generation=1, observed_at_unix=1.0)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        app.state.store.get = AsyncMock(return_value=dict(kind="stream", media_failure=failure,
                                                         source_ref="rtsp://private/?token=secret"))
        path = '/v1/tasks/00000000-0000-0000-0000-000000000001/diagnostics'
        response = await client.get(path)
        assert response.status_code == 200
        assert response.json() == {"mode": "stream", "last_failure": failure}
        assert 'secret' not in response.text
        assert (await client.get(path + '?sample=64')).status_code == 409


@pytest.mark.parametrize("headers", [
    {"X-Detection-Interval": "2"},
    {"X-Tracking-Mode": "flow"},
    {"X-Detection-Interval": "1", "X-Tracking-Mode": "flow"},
    {"X-Detection-Interval": "3", "X-Tracking-Mode": "invalid"},
])
async def test_invalid_stream_cadence_is_rejected_before_admission(tmp_path, monkeypatch, headers):
    from unittest.mock import AsyncMock

    app = create_app(Settings(_env_file=None, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        create = AsyncMock()
        monkeypatch.setattr(app.state.store, "create", create)
        response = await client.post("/v1/stream-tasks", json={"url": "rtsp://127.0.0.1/source"}, headers=headers)
        assert response.status_code == 422
        create.assert_not_awaited()
