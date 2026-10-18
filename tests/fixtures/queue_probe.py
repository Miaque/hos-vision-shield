"""Minimal adapter under test; promote it to application code only after T009."""

import asyncio
import os

from redis.asyncio import Redis
from taskiq import AckableMessage, Context, TaskiqDepends
from taskiq_redis import RedisStreamBroker

from vision_shield.jobs.broker import BoundedRedisBroker, ENQUEUE, ACK_DELETE


class BoundedProbeBroker(BoundedRedisBroker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, message_limit=2, **kwargs)


broker = BoundedProbeBroker(
    os.getenv("TEST_PROBE_REDIS_URL", "redis://127.0.0.1:16379/0"),
    queue_name="vs:probe",
    consumer_group_name="vs:probe",
    consumer_id="0",
    xread_count=1,
    xread_block=100,
    unacknowledged_lock_timeout=5,
)


@broker.task
async def long_probe(logical_id: str, context: Context = TaskiqDepends()):
    async with Redis(connection_pool=broker.connection_pool) as client:
        if not await client.set(f"vs:owner:{logical_id}", "owned", nx=True):
            await broker.ack_message(context.message.task_id)
            await client.incr("vs:duplicate_count")
            return
        await broker.ack_message(context.message.task_id)
        await client.incr("vs:started_count")
        while not await client.exists("vs:release"):
            await asyncio.sleep(0.05)
        await client.incr("vs:finished_count")
