import httpx
import pytest
from redis.asyncio import Redis

from vision_shield.config import Settings
from vision_shield.main import create_app
from vision_shield.jobs import tasks


@pytest.mark.parametrize("publish_url", [None, "rtsp://127.0.0.1/custom?token=secret", "rtmp://127.0.0.1/live/custom?sign=secret"])
async def test_stream_intent_is_stoppable_without_a_file_result(redis_url, tmp_path, publish_url):
    app = create_app(Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        redis = app.state.redis
        await redis.flushdb()
        seconds, micros = await redis.time()
        await redis.zadd("vs:workers:streams", {"test": seconds * 1000 + micros // 1000})
        body = {"url": "rtsp://127.0.0.1/source"}
        if publish_url:
            body["publish_url"] = publish_url
        created = await client.post("/v1/stream-tasks", json=body, headers={"Idempotency-Key": "stream"})
        assert created.status_code == 202
        task = created.json()
        assert bool(task["playback_url"]) is (publish_url is None) and not task["ready"]
        assert "secret" not in created.text
        stored = await app.state.store.get(task["id"])
        assert stored.get("publish_url") == publish_url
        assert (await client.post("/v1/stream-tasks", json=body, headers={"Idempotency-Key": "stream"})).json()["id"] == task["id"]
        changed = body | {"publish_url": "rtmp://127.0.0.1/live/other"}
        assert (await client.post("/v1/stream-tasks", json=changed, headers={"Idempotency-Key": "stream"})).status_code == 409
        assert (await client.get(f"/v1/tasks/{task['id']}/result")).status_code == 409
        for _ in range(2):
            stopped = await client.patch(f"/v1/tasks/{task['id']}", json={"desired_state": "stopped"})
            assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"


@pytest.mark.parametrize("scheme", ["rtsp", "rtmp"])
async def test_direct_publisher_stop_uses_process_exit_confirmation(redis_url, tmp_path, monkeypatch, scheme):
    import asyncio
    import time

    settings = Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path,
                        source_allowlist=["127.0.0.1"], mediamtx_control_url="http://127.0.0.1:1")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await app.state.redis.flushdb()
        store = app.state.store
        target = f"{scheme}://127.0.0.1/app/camera?sign=secret"
        task = await store.create("stream", "rtsp", "rtsp://127.0.0.1/source", publish_url=target)
        owned = await store.claim(task["id"], "owner")
        cancel = asyncio.Event()

        async def run(command, payload, cancel, on_event, timeout, *, on_exit=None):
            assert payload["publish_url"] == target and payload["config"]["publish_url"] == target
            await on_event(dict(output=True, output_monotonic=time.monotonic(), ready=True))
            current = await store.get(task["id"])
            assert current["ready"] and current["state"] == "running"
            await store.request_stop(task["id"], "stopped")
            cancel.set()
            if on_exit:
                await on_exit()
            return dict(cancelled=True, returncode=0)

        monkeypatch.setattr(tasks, "run_process", run)
        await tasks.run_stream(owned, store, settings, cancel, "owner", owned["generation"])
        current = await store.get(task["id"])
        assert current["state"] == "stopped" and not current["publish_allowed"]
        assert current["publish_url"] == target and not current["sessions"]
