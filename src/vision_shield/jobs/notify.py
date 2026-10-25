"""独立通知进程：有界领取、MQ 发布确认、持久退避和可接管租约。"""

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
from datetime import datetime, timedelta, timezone

from aio_pika.exceptions import AMQPException
from redis.asyncio import Redis
from redis.exceptions import RedisError

from vision_shield.api.schemas import StreamHealth
from vision_shield.config import Settings
from vision_shield.jobs.amqp_notifications import AmqpNotifications
from vision_shield.jobs.store import TaskStore, stream_health

logger = logging.getLogger(__name__)
SAFE_ERRORS = frozenset({"worker_lost", "processing_failed", "storage_full", "source_unreachable",
                         "media_too_large", "invalid_media", "unsupported_media", "invalid_source",
                         "source_not_allowed", "download_failed", "source_timeout"})


def notification_body(task: dict) -> bytes:
    """只序列化状态合同，不携带输入、输出地址或内部错误文本。"""
    health = stream_health(task, task["_observed_ms"])
    error = task.get("error")
    error_code = error.get("code") if error else None
    if error_code is not None and error_code not in SAFE_ERRORS:
        error_code = "processing_failed"
    occurred = datetime.fromisoformat(task["updated_at"]).astimezone(timezone(timedelta(hours=8)))
    body = dict(schema_version=1, event_id=f"{task['id']}:{task['sequence']}",
                sequence=task["sequence"], task_id=task["id"], submission_key=task["submission_key"],
                kind=task["kind"], state=task["state"], desired_state=task["desired_state"],
                ready=task["ready"], occurred_at=occurred.strftime("%Y-%m-%d %H:%M:%S"),
                health=StreamHealth(**health).model_dump(mode="json") if health else None,
                error_code=error_code)
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()


async def deliver(store: TaskStore, publisher: AmqpNotifications, settings: Settings, event: dict) -> None:
    """持久发布确认后交接责任，失败只重试通知，不重跑推理。"""
    target_id = event["task"]["notification_target"]
    target = settings.notification_targets[target_id]
    body = notification_body(event["task"])
    delivered = False
    try:
        async with asyncio.timeout(5):
            await publisher.publish(target_id, body, target.secret.get_secret_value(),
                                    f"{event['task']['id']}:{event['task']['sequence']}")
        delivered = True
    except (AMQPException, OSError, TimeoutError, RuntimeError):
        logger.warning("通知未确认 target=%s task=%s attempts=%s", target_id,
                       event["task"]["id"], event["attempts"] + 1)
    delay = min(300, 2 ** min(event["attempts"], 9) * random.uniform(1, 1.25))
    await store.complete_notification(event, delivered=delivered, delay_seconds=delay)


async def main():
    """独立运行通知派发，多个实例共享 Redis 领取与并发上限。"""
    settings = Settings()
    current = asyncio.current_task()
    if os.name != "nt":
        for signum in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(signum, current.cancel)
    redis = Redis.from_url(settings.redis_url.get_secret_value(), socket_connect_timeout=2, socket_timeout=2)
    store = TaskStore(redis)
    publisher = AmqpNotifications(settings.vision_shield_notification_amqp_url.get_secret_value())
    targets = tuple(settings.notification_targets)
    try:
        while True:
            try:
                events = await store.claim_notifications(targets=targets)
                targets = targets[1:] + targets[:1]
                results = await asyncio.gather(*(deliver(store, publisher, settings, event) for event in events),
                                               return_exceptions=True)
                if any(isinstance(result, Exception) for result in results):
                    logger.error("通知派发异常，持久责任保留等待租约恢复")
            except RedisError:
                logger.error("通知存储不可用，等待恢复")
            await asyncio.sleep(0.5)
    finally:
        await publisher.close()
        await redis.aclose()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(main())
