"""MQ 投递必须收到发布确认，环境目标参与签名。"""

import hashlib
import hmac
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from vision_shield.config import Settings
from vision_shield.jobs.amqp_notifications import AmqpNotifications, queue_name


def test_amqp_targets_require_valid_ids_and_connection_and_reject_http():
    """通用目标来自部署配置，MQ 连接不能默默退化为 HTTP。"""
    settings = Settings(_env_file=None, vision_shield_notification_amqp_url="amqp://mq/%2F",
                        notification_targets={"dev.platform": {"secret": "s" * 32}})
    assert settings.notification_targets["dev.platform"].secret.get_secret_value() == "s" * 32
    assert queue_name("dev.platform") == "hos.vision-shield.task-events.v1.dev.platform"
    for values in (
        {"notification_targets": {"dev.platform": {"secret": "s" * 32}}},
        {"notification_targets": {"../bad": {"url": "http://platform/callback", "secret": "s" * 32}}},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **values)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vision_shield_notification_amqp_url="amqp://mq/%2F", notification_targets={
            "platform": {"url": "http://platform/callback", "secret": "s" * 32}})



@pytest.mark.asyncio
async def test_publisher_uses_persistent_message_target_signature_and_confirm(monkeypatch):
    """只在 broker 确认后返回，发布异常保留上层 Redis 通知责任。"""
    from pamqp.commands import Basic

    from vision_shield.jobs import amqp_notifications as module

    channel = SimpleNamespace(declare_queue=AsyncMock(), default_exchange=SimpleNamespace(
        publish=AsyncMock(return_value=Basic.Ack(delivery_tag=1))), is_closed=False)
    connection = SimpleNamespace(channel=AsyncMock(return_value=channel), close=AsyncMock(), is_closed=False)
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(module.aio_pika, "connect_robust", connect)
    publisher = AmqpNotifications("amqp://mq/%2F")
    body = b'{"event_id":"task:2"}'
    await publisher.publish("dev.platform", body, "s" * 32, "task:2")
    message = channel.default_exchange.publish.call_args.args[0]
    assert message.body == body and message.delivery_mode == 2
    assert message.headers["X-Vision-Shield-Target"] == "dev.platform"
    assert message.headers["X-Vision-Shield-Signature"] == hmac.new(
        b"s" * 32, b"dev.platform\n" + body, hashlib.sha256).hexdigest()
    assert channel.default_exchange.publish.call_args.kwargs["mandatory"] is True
    assert channel.declare_queue.call_args.args[0] == queue_name("dev.platform")
    channel.default_exchange.publish.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await publisher.publish("dev.platform", body, "s" * 32, "task:2")
    await publisher.close()
    connection.close.assert_awaited_once()
