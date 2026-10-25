"""默认 vhost 中使用独立队列，只在持久发布确认后交接通知责任。"""

import asyncio

import aio_pika
from pamqp.commands import Basic

QUEUE_PREFIX = "hos.vision-shield.task-events.v1."
QUEUE_ARGUMENTS = {"x-queue-type": "quorum", "x-max-length": 10000, "x-overflow": "reject-publish",
                   "x-delivery-limit": -1}


def queue_name(target: str) -> str:
    """目标已由配置白名单校验，不复用任何业务队列。"""
    return QUEUE_PREFIX + target


class AmqpNotifications:
    """连接按 notifier 生命周期复用；局部锁只保护可重建的连接资源。"""

    def __init__(self, url: str):
        """保存部署连接，首次 MQ 投递时建立连接。"""
        self.url = url
        self.connection = None
        self.channel = None
        self.queues = set()
        self.lock = asyncio.Lock()

    async def publish(self, target: str, body: bytes, event_id: str) -> None:
        """按已登记目标投递；超时、无路由、负确认均保留 Redis 重试责任。"""
        async with self.lock:
            if self.connection is None or self.connection.is_closed or self.channel is None or self.channel.is_closed:
                await self.close()
                self.connection = await aio_pika.connect_robust(self.url, timeout=5)
                self.channel = await self.connection.channel(publisher_confirms=True, on_return_raises=True)
                self.queues.clear()
            if target not in self.queues:
                await self.channel.declare_queue(queue_name(target), durable=True, arguments=QUEUE_ARGUMENTS, timeout=5)
                self.queues.add(target)
            channel = self.channel
        result = await channel.default_exchange.publish(aio_pika.Message(
            body, content_type="application/json", delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=event_id, headers={"X-Vision-Shield-Target": target}),
            routing_key=queue_name(target), mandatory=True, timeout=5)
        if result is None or not isinstance(result, Basic.Ack):
            raise RuntimeError("通知未收到 RabbitMQ 发布确认")

    async def close(self) -> None:
        """退出时关闭连接及其 channel；通知责任仍在 Redis 或持久队列。"""
        if self.connection is not None:
            await self.connection.close()
            self.connection, self.channel = None, None
            self.queues.clear()
