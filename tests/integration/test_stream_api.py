import httpx
from redis.asyncio import Redis

from vision_shield.config import Settings
from vision_shield.main import create_app


async def test_stream_intent_is_stoppable_without_a_file_result(redis_url, tmp_path):
    app = create_app(Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        redis = app.state.redis
        await redis.flushdb()
        seconds, micros = await redis.time()
        await redis.zadd("vs:workers:streams", {"test": seconds * 1000 + micros // 1000})
        created = await client.post("/v1/stream-tasks", json={"url": "rtsp://127.0.0.1/source"})
        assert created.status_code == 202
        task = created.json()
        assert task["playback_url"] and not task["ready"]
        assert (await client.get(f"/v1/tasks/{task['id']}/result")).status_code == 409
        for _ in range(2):
            stopped = await client.patch(f"/v1/tasks/{task['id']}", json={"desired_state": "stopped"})
            assert stopped.status_code == 200 and stopped.json()["state"] == "stopped"
