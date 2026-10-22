from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.api.schemas import TaskResponse
from vision_shield.config import Settings
from vision_shield.main import create_app


def task_record(kind="video"):
    return dict(id="00000000-0000-0000-0000-000000000001", kind=kind, state="queued",
                desired_state="run", ready=False, created_at="2026-10-21T20:15:30.999999Z",
                updated_at="2026-10-22T12:15:31+05:30", error=None, stats={})


@pytest.mark.parametrize("stored,expected", [
    ("2026-10-21T20:15:30.999999Z", "2026-10-22 04:15:30"),
    ("2026-10-22T12:15:30+08:00", "2026-10-22 12:15:30"),
    ("2026-12-31T18:00:00-04:00", "2027-01-01 06:00:00"),
])
def test_task_dates_convert_to_utc8_without_changing_stored_records(stored, expected):
    record = task_record()
    record.update(created_at=stored, updated_at=stored)
    body = TaskResponse.from_record(record, "rtsp://media:8554").model_dump(mode="json")
    assert body["created_at"] == body["updated_at"] == expected
    assert record["created_at"] == record["updated_at"] == stored


@pytest.mark.parametrize("operation", ["file", "upload", "stream", "detail", "list", "stop"])
async def test_all_task_endpoints_use_the_same_datetime_contract(operation, tmp_path, monkeypatch):
    from vision_shield.api import routes

    monkeypatch.setattr(routes, "require_worker", AsyncMock())
    monkeypatch.setattr(routes, "validate_source", AsyncMock())
    record = task_record("stream" if operation == "stream" else "video")
    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        app.state.store.create = AsyncMock(return_value=record)
        app.state.store.get = AsyncMock(return_value=record)
        app.state.store.active = AsyncMock(return_value=[record])
        app.state.store.request_stop = AsyncMock(return_value=record)
        path = f"/v1/tasks/{record['id']}"
        if operation == "file":
            response = await client.post("/v1/file-tasks", json={"kind": "video", "url": "http://camera/a.mp4"})
        elif operation == "upload":
            response = await client.post("/v1/file-tasks/uploads", data={"kind": "video"},
                                         files={"file": ("a.mp4", b"video", "video/mp4")})
        elif operation == "stream":
            response = await client.post("/v1/stream-tasks", json={"url": "rtsp://camera/live"})
        elif operation == "stop":
            response = await client.patch(path, json={"desired_state": "cancelled"})
        else:
            response = await client.get("/v1/tasks" if operation == "list" else path)
        assert response.status_code in (200, 202)
        body = response.json()["queued"][0] if operation == "list" else response.json()
        assert body["created_at"] == "2026-10-22 04:15:30"
        assert body["updated_at"] == "2026-10-22 14:45:31"


def test_openapi_describes_utc8_strings_instead_of_iso8601_dates():
    schema = create_app(Settings(_env_file=None)).openapi()["components"]["schemas"]["TaskResponse"]
    for name in ("created_at", "updated_at"):
        field = schema["properties"][name]
        assert field["type"] == "string" and "format" not in field
        assert "UTC+8" in field["description"]
        assert field["examples"] == ["2026-10-22 09:00:00"]
        assert "pattern" in field
