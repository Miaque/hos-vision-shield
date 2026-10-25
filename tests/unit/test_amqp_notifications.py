"""MQ 投递必须收到发布确认，环境目标决定队列。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from vision_shield.config import Settings
from vision_shield.jobs.amqp_notifications import AmqpNotifications, QUEUE_ARGUMENTS, queue_name


def test_amqp_targets_require_valid_ids_and_connection_and_reject_http():
    """通用目标来自部署配置，MQ 连接不能默默退化为 HTTP。"""
    settings = Settings(_env_file=None, vision_shield_notification_amqp_url="amqp://mq/%2F",
                        notification_targets=["dev.platform", "prod.event-hub"])
    assert settings.notification_targets == ["dev.platform", "prod.event-hub"]
    assert queue_name("dev.platform") == "hos.vision-shield.task-events.v1.dev.platform"
    assert Settings(_env_file=None).notification_targets == []
    for values in (
        {"notification_targets": ["dev.platform"]},
        {"notification_targets": ["../bad"]},
        {"notification_targets": {"platform": {"url": "http://platform/callback"}}},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **values)



def test_queue_declaration_round_trips_unlimited_redelivery_argument():
    """真实 AMQP 编解码必须保留 -1，不能只用 mock 验证队列声明。"""
    from pamqp import frame
    from pamqp.commands import Queue

    declaration = Queue.Declare(queue=queue_name("dev.platform"), durable=True, arguments=QUEUE_ARGUMENTS)
    encoded = frame.marshal(declaration, 1)
    consumed, channel, decoded = frame.unmarshal(encoded)
    assert consumed == len(encoded) and channel == 1
    assert decoded.queue == declaration.queue and decoded.durable
    assert decoded.arguments == QUEUE_ARGUMENTS
    assert decoded.arguments["x-delivery-limit"] == -1


@pytest.mark.asyncio
async def test_publisher_uses_persistent_message_target_and_confirm(monkeypatch):
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
    await publisher.publish("dev.platform", body, "task:2")
    message = channel.default_exchange.publish.call_args.args[0]
    assert message.body == body and message.delivery_mode == 2
    assert message.headers == {"X-Vision-Shield-Target": "dev.platform"}
    assert channel.default_exchange.publish.call_args.kwargs["mandatory"] is True
    assert channel.declare_queue.call_args.args[0] == queue_name("dev.platform")
    channel.default_exchange.publish.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await publisher.publish("dev.platform", body, "task:2")
    await publisher.close()
    connection.close.assert_awaited_once()
