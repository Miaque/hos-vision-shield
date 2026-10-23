"""通知签名、安全投影和健康期限的无设施测试。"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.config import Settings
from vision_shield.jobs.notify import deliver, notification_body
from vision_shield.jobs.store import stream_health


def record():
    """固定状态记录不含任何真实凭据。"""
    return dict(id="00000000-0000-4000-8000-000000000001", kind="video", sequence=7,
                state="succeeded", desired_state="run", ready=True, submission_key="hub:1",
                notification_target="event-hub", updated_at="2026-10-22T00:00:00+00:00",
                _observed_ms=100000, error=None, source_ref="private-secret", result_ref="private-result")


@pytest.mark.parametrize("status", [202, 401, 410, 503, 302])
async def test_notification_signature_and_confirmation(status):
    """只有 2xx 交接责任，重定向不能把签名发送给其他地址。"""
    secret = "x" * 32
    settings = Settings(_env_file=None, notification_targets={"event-hub": {
        "url": "http://hub/internal/events", "secret": secret}})
    def respond(request):
        timestamp = request.headers["X-Vision-Shield-Timestamp"]
        expected = hmac.new(secret.encode(), timestamp.encode() + b"." + request.content,
                            hashlib.sha256).hexdigest()
        assert hmac.compare_digest(request.headers["X-Vision-Shield-Signature"], expected)
        assert b"private-secret" not in request.content and b"private-result" not in request.content
        assert json.loads(request.content)["event_id"].endswith(":7")
        return httpx.Response(status, headers={"Location": "http://untrusted"})
    store = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await deliver(store, client, settings, dict(task=record(), token="token", attempts=0))
    assert store.complete_notification.await_args.kwargs["delivered"] is (status == 202)
    assert 1 <= store.complete_notification.await_args.kwargs["delay_seconds"] <= 300


def test_untrusted_error_text_never_escapes():
    """内部异常或来源信息不能通过错误字段旁路。"""
    task = record() | {"error": {"code": "private-secret", "message": "private-secret"}}
    assert json.loads(notification_body(task))["error_code"] == "processing_failed"


def test_health_never_renews_from_observation_or_stale_owner():
    """查询时刻仅用于判定租约，不成为输出时间。"""
    task = record() | dict(kind="stream", owner_id="worker", generation=3,
                          last_output_ms=10000, lease_until=100000)
    first, late = stream_health(task, 11000), stream_health(task, 60000)
    assert first["valid_until"] == late["valid_until"]
    assert first["valid_until"] == "1970-01-01T00:00:55+00:00"
    assert stream_health(task, 100001)["valid_until"] is None
    assert stream_health(task | {"last_output_ms": None}, 11000)["valid_until"] is None


def test_notification_target_rejects_arbitrary_destinations_and_short_secrets():
    """目标由部署注册，签名密钥有最小长度。"""
    for target in ({"unknown": {"url": "http://hub", "secret": "x" * 32}},
                   {"event-hub": {"url": "http://user:password@hub", "secret": "x" * 32}},
                   {"event-hub": {"url": "http://hub", "secret": "short"}}):
        with pytest.raises(ValueError):
            Settings(_env_file=None, notification_targets=target)
