"""创建合同的无设施边界测试。"""

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
                       ("notification_target", "platform"), ("profile_detection", True)):
        assert digest != request_fingerprint(**(values | {key: value}))
    assert request_fingerprint(**(values | {'profile_detection': 'host'})) not in (
        digest, request_fingerprint(**(values | {'profile_detection': True})))


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
