"""通知交付、安全投影和健康期限的无设施测试。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


@pytest.mark.parametrize("failure", [None, TimeoutError(), RuntimeError("negative confirmation")])
async def test_notification_confirms_only_after_persistent_publish(failure):
    """发布失败保留重试责任，通知不包含来源或成品凭据。"""
    publisher = AsyncMock()
    publisher.publish.side_effect = failure
    store = AsyncMock()
    await deliver(store, publisher, {"task": record(), "token": "token", "attempts": 0})
    body = publisher.publish.await_args.args[1]
    assert b"private-secret" not in body and b"private-result" not in body
    assert json.loads(body)["event_id"].endswith(":7")
    assert store.complete_notification.await_args.kwargs["delivered"] is (failure is None)
    assert 1 <= store.complete_notification.await_args.kwargs["delay_seconds"] <= 300


async def test_reconnect_wait_is_bounded_and_retains_notification(monkeypatch):
    """即使 AMQP 库在重连前等待，Redis 通知租约也不能被永久占用。"""
    import asyncio

    from vision_shield.jobs import notify
    original_timeout = asyncio.timeout
    def bounded(seconds):
        assert seconds == 5
        return original_timeout(0.02)
    monkeypatch.setattr(notify.asyncio, "timeout", bounded)
    publisher, store = AsyncMock(), AsyncMock()
    async def disconnected(*args):
        await asyncio.Event().wait()
    publisher.publish.side_effect = disconnected
    await deliver(store, publisher, {"task": record(), "token": "token", "attempts": 0})
    assert store.complete_notification.await_args.kwargs["delivered"] is False


@pytest.mark.parametrize("failure_at", ["body", "publish", "complete"])
async def test_dispatch_failure_logs_safe_event_context_and_keeps_other_deliveries(monkeypatch, caplog, failure_at):
    """派发异常可定位到事件和调用位置，但不泄露异常文本中的凭据。"""
    from redis.exceptions import ConnectionError

    from vision_shield.jobs import notify

    failed = {"task": record(), "token": "token", "attempts": 2}
    healthy = {"task": record() | {"id": "other-task"}, "token": "other", "attempts": 0}
    publisher, store, client = AsyncMock(), AsyncMock(), AsyncMock()
    store.claim_notifications.side_effect = [[healthy, failed], [], asyncio.CancelledError()]
    error_type = "ConnectionError" if failure_at == "complete" else "ValueError"
    failure_site = {"body": "notification_body", "publish": "publish", "complete": "complete"}[failure_at]
    if failure_at == "body":
        failed["task"]["updated_at"] = "private-secret"

    async def publish(target, body, event_id):
        if failure_at == "publish" and event_id.startswith(failed["task"]["id"]):
            raise ValueError("private-secret")

    async def complete(event, **kwargs):
        if failure_at == "complete" and event is failed:
            raise ConnectionError("private-secret")
        return True

    publisher.publish.side_effect = publish
    store.complete_notification.side_effect = complete
    monkeypatch.setattr(notify, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(notify, "Redis", SimpleNamespace(from_url=lambda *a, **kw: client))
    monkeypatch.setattr(notify, "TaskStore", lambda *a: store)
    monkeypatch.setattr(notify, "AmqpNotifications", lambda *a: publisher)
    monkeypatch.setattr(notify.asyncio, "sleep", AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await notify.main()

    logs = [entry.getMessage() for entry in caplog.records if "通知派发异常" in entry.getMessage()]
    assert len(logs) == 1
    assert f"target=event-hub task={failed['task']['id']} sequence=7 attempts=3" in logs[0]
    assert f"type={error_type}" in logs[0] and f"{failure_site}:" in logs[0]
    assert "private-secret" not in caplog.text and "private-result" not in caplog.text
    assert store.claim_notifications.await_count == 3
    assert any(call.args[0] is healthy and call.kwargs["delivered"]
               for call in store.complete_notification.await_args_list)
    if failure_at != "complete":
        assert not any(call.args[0] is failed for call in store.complete_notification.await_args_list)
    publisher.close.assert_awaited_once()
    client.aclose.assert_awaited_once()


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


def test_notification_target_rejects_arbitrary_destinations():
    """仅接受合法目标名，不允许通知目标携带任意回调 URL。"""
    for target in (["http://hub"], ["../hub"], [""]):
        with pytest.raises(ValueError):
            Settings(_env_file=None, vision_shield_notification_amqp_url="amqp://mq/%2F",
                     notification_targets=target)
