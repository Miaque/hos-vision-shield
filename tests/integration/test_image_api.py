from uuid import uuid4

import httpx
import pytest
from redis.asyncio import Redis

from vision_shield.config import Settings
from vision_shield.main import create_app


async def test_file_contract_and_cancel_before_execution(redis_url, tmp_path):
    settings = Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path,
                        source_allowlist=["127.0.0.1"], file_task_limit=1)
    app = create_app(settings)
    schema = app.openapi()["paths"]["/v1/file-tasks/uploads"]["post"]["requestBody"]
    assert schema["content"]["multipart/form-data"]["schema"]["required"] == ["kind", "file"]
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        redis = app.state.redis
        await redis.flushdb()
        seconds, micros = await redis.time()
        await redis.zadd("vs:workers:files", {"test": seconds * 1000 + micros // 1000})
        result = await client.post("/v1/file-tasks", json={"kind": "image", "url": "http://127.0.0.1/a.png"})
        assert result.status_code == 202
        task = result.json()
        assert result.headers["location"] == f"/v1/tasks/{task['id']}"
        assert "source_ref" not in task and task["result_url"] is None
        assert (await client.get(result.headers["location"])).status_code == 200
        assert (await client.get(f"/v1/tasks/{task['id']}/result")).status_code == 409
        assert (await client.post("/v1/file-tasks", json={"kind": "image", "url": "http://127.0.0.1/b.png"})).status_code == 429
        for _ in range(2):
            cancelled = await client.patch(result.headers["location"], json={"desired_state": "cancelled"})
            assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled"
        assert (await client.get(f"/v1/tasks/{uuid4()}")).status_code == 404
        assert (await client.post("/v1/file-tasks", json={"kind": "face", "url": "file:///secret"})).json()["error"]["code"] == "invalid_input"
