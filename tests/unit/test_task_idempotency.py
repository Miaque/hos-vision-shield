"""创建合同的无设施边界测试。"""

import hashlib
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.api.schemas import TaskQueryInput
from vision_shield.config import Settings
from vision_shield.main import create_app
from vision_shield.jobs.store import request_fingerprint


def test_fingerprint_tracks_content_and_effective_parameters():
    """上传随机路径不参与指纹，有效参数和内容必须参与。"""
    values = dict(kind="video", source_kind="upload", source_digest="abc", redaction_mode="solid",
                  notification_target="event-hub", profile_detection=False)
    digest = request_fingerprint(**values)
    assert digest == request_fingerprint(**dict(reversed(list(values.items()))))
    for key, value in (("source_digest", "def"), ("redaction_mode", "mosaic"),
                       ("notification_target", "platform"), ("max_bitrate", 20_000_000), ("profile_detection", True),
                       ("profile_detection", "host"), ("profile_detection", "forward-primary-face")):
        assert digest != request_fingerprint(**(values | {key: value}))


def test_batch_is_bounded_and_rejects_invalid_identifiers():
    """批量查询入口必须有明确上限，不能退化成全表读取。"""
    identifier = "00000000-0000-4000-8000-000000000001"
    assert len(TaskQueryInput(task_ids=[identifier]).task_ids) == 1
    for ids in ([], [identifier] * 101, ["invalid"]):
        with pytest.raises(ValueError):
            TaskQueryInput(task_ids=ids)


@pytest.mark.parametrize("headers", [
    {"X-Notification-Target": "event-hub"},
    {"Idempotency-Key": "key", "X-Notification-Target": "http://untrusted"},
    {"Idempotency-Key": " "},
    {"Idempotency-Key": "x" * 129},
])
async def test_invalid_notification_headers_fail_before_external_io(headers):
    """不允许调用方指定回调地址或绕过稳定意图。"""
    app = create_app(Settings(_env_file=None))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        app.state.store.create = AsyncMock()
        response = await client.post("/v1/stream-tasks", headers=headers,
                                     json={"url": "rtsp://camera/live"})
        assert response.status_code == 422
        app.state.store.create.assert_not_awaited()


@pytest.mark.parametrize("headers", [
    {},
    {"X-Detection-Interval": "1", "X-Tracking-Mode": "template"},
    {"X-Detection-Interval": "3"},
    {"X-Detection-Interval": "3", "X-Tracking-Mode": "template"},
    {"X-Detection-Interval": "3", "X-Tracking-Mode": "flow"},
])
async def test_upload_fingerprint_includes_effective_cadence_and_tracking(tmp_path, monkeypatch, headers):
    """新参数参与幂等判断，显式默认值仍兼容既有指纹。"""
    from vision_shield.api import routes

    monkeypatch.setattr(routes, "require_worker", AsyncMock())
    app = create_app(Settings(_env_file=None, media_root=tmp_path,
        notification_targets={"event-hub": {"url": "http://hub/internal", "secret": "x" * 32}}))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        app.state.store.replay = AsyncMock(return_value=None)
        app.state.store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit: record)
        response = await client.post("/v1/file-tasks/uploads",
            headers={"Idempotency-Key": "hub:cadence", "X-Notification-Target": "event-hub", **headers},
            data={"kind": "video"}, files={"file": ("input.mp4", b"video")})
        assert response.status_code == 202
        record = app.state.store._change.call_args.args[2]
        options = {}
        if headers.get("X-Detection-Interval") == "3":
            options["detection_interval"] = 3
        if headers.get("X-Tracking-Mode") == "flow":
            options["tracking"] = "flow"
        expected = request_fingerprint(kind="video", source_kind="upload",
            source_digest=hashlib.sha256(b"video").hexdigest(), redaction_mode="solid",
            notification_target="event-hub", profile_detection=False, **options)
        assert record["fingerprint"] == expected
        assert record["submission_key"] == "hub:cadence"
        assert record["notification_target"] == "event-hub"
        assert record.get("detection_interval", 1) == int(headers.get("X-Detection-Interval", "1"))
        assert record.get("tracking", "template") == headers.get("X-Tracking-Mode", "template")
