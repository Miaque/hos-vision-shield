import json
from pathlib import Path

import httpx
import pytest
from redis.asyncio import Redis
from redis.exceptions import OutOfMemoryError

from vision_shield.config import Settings
from vision_shield.jobs.store import TaskStore
from vision_shield.main import create_app
from vision_shield.media.publisher import Publisher


async def test_redis_full_does_not_create_partial_task(redis_url):
    async with Redis.from_url(redis_url) as client:
        await client.flushdb()
        try:
            await client.config_set("maxmemory", 1)
            with pytest.raises(OutOfMemoryError):
                await TaskStore(client).create("image", "upload", "private.bin")
        finally:
            await client.config_set("maxmemory", 0)
        assert not await client.smembers("vs:active:files")


async def test_disk_failure_body_limits_and_private_paths(redis_url, tmp_path, monkeypatch, caplog):
    app = create_app(Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        await app.state.redis.flushdb()
        seconds, micros = await app.state.redis.time()
        await app.state.redis.zadd("vs:workers:files", {"test": seconds * 1000 + micros // 1000})
        original_open = Path.open

        def fail_write(path, *args, **kwargs):
            if path.name == "source.bin":
                raise OSError(28, "No space left on device")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_write)
        response = await client.post("/v1/file-tasks/uploads", data={"kind": "image"}, files={"file": ("../../secret", b"media")})
        assert response.status_code == 500 and "secret" not in response.text
        assert not await app.state.redis.smembers("vs:active:files")
        assert (await client.post("/v1/file-tasks", content=b"x" * 65537)).status_code == 413
        response = await client.post("/v1/file-tasks", json={"kind": "image", "url": "http://user:secret@localhost/a"})
        assert response.status_code == 422 and "secret" not in response.text + caplog.text
        task = await app.state.store.create("image", "upload", "private")
        owned = await app.state.store.claim(task["id"], "worker")
        await app.state.store.finish(task["id"], "worker", owned["generation"], "succeeded", "../private.txt")
        assert (await client.get(f"/v1/tasks/{task['id']}/result")).status_code == 409


async def test_control_unavailable_keeps_stop_pending_and_gate_closed(redis_url):
    async with Redis.from_url(redis_url) as client:
        await client.flushdb()
        store = TaskStore(client)
        task = await store.create("stream", "rtsp", "rtsp://camera/live")
        owned = await store.claim(task["id"], "worker")
        publisher = Publisher(store, Settings(_env_file=None, mediamtx_control_url="http://127.0.0.1:1"))
        await publisher._change(task["id"], "open", owner_id="worker", generation=owned["generation"])
        assert await publisher.authorize(dict(user=f"{task['id']}.{owned['generation']}", password="worker",
                                               action="publish", path=f"redacted/{task['id']}", id="test-session"))
        await store.request_stop(task["id"], "stopped")
        with pytest.raises(httpx.HTTPError):
            await publisher.close(task)
        current = await store.get(task["id"])
        assert current["state"] != "stopped" and not current["publish_allowed"]
        assert current["sessions"] == {"test-session": owned["generation"]}
