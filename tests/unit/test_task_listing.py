from unittest.mock import AsyncMock

import httpx

from vision_shield.config import Settings
from vision_shield.main import create_app


async def test_task_listing_groups_active_tasks_and_hides_internal_fields():
    records = []
    cases = [
        ("queued", "image", None, "run"),
        ("queued", "video", None, "run"),
        ("queued", "stream", None, "run"),
        ("processing", "video", "worker", "cancelled"),
        ("connecting", "stream", "worker", "run"),
        ("running", "stream", "worker", "run"),
        ("recovering", "stream", "worker", "run"),
        ("recovering", "stream", None, "run"),
        ("recovering", "stream", None, "stopped"),
        ("succeeded", "image", None, "run"),
        ("failed", "video", None, "run"),
        ("cancelled", "image", None, "cancelled"),
        ("stopped", "stream", None, "stopped"),
    ]
    for index, (state, kind, owner, desired) in enumerate(cases):
        records.append(dict(
            id=f"00000000-0000-0000-0000-{index:012d}", state=state, kind=kind,
            owner_id=owner, desired_state=desired, ready=False,
            created_at=f"2026-10-21T00:00:{index:02d}+00:00",
            updated_at="2026-10-21T00:01:00+00:00", error=None, stats={},
            source_ref="rtsp://user:password@camera/live", result_ref=None,
        ))
    app = create_app(Settings(_env_file=None))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        app.state.store.active = AsyncMock(return_value=list(reversed(records)))
        response = await client.get("/v1/tasks")
        assert response.status_code == 200
        body = response.json()
        assert [task["id"] for task in body["queued"]] == [records[i]["id"] for i in (0, 1, 2, 7)]
        assert [task["id"] for task in body["current"]] == [records[i]["id"] for i in (3, 4, 5, 6, 8)]
        for task in body["current"] + body["queued"]:
            assert not {"source_ref", "owner_id", "result_ref"} & task.keys()
            assert "stats" in task and "playback_url" in task
        app.state.store.get = AsyncMock(return_value=records[0])
        detail = await client.get(f"/v1/tasks/{records[0]['id']}")
        assert detail.json() == body["queued"][0]
        app.state.store.active.return_value = []
        assert (await client.get("/v1/tasks")).json() == {"current": [], "queued": []}


async def test_task_listing_reports_redis_unavailable():
    app = create_app(Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0"))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/tasks")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dependency_unavailable"
