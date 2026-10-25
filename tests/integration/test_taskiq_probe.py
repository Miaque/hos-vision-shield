import asyncio
import os
import subprocess
import sys

import pytest
from redis.asyncio import BlockingConnectionPool, Redis

from tests.conftest import NO_WINDOW, ROOT, stop_process
from tests.fixtures.queue_probe import ACK_DELETE, ENQUEUE

pytestmark = pytest.mark.integration


async def wait_value(client, key, value):
    async with asyncio.timeout(15):
        while await client.get(key) != value:
            await asyncio.sleep(0.05)


async def test_bounded_enqueue_keeps_pending_and_ack_deletes_atomically(redis_url):
    async with Redis.from_url(redis_url) as client:
        stream = "vs:atomic-probe"
        await client.xgroup_create(stream, "probe", id="0", mkstream=True)
        first = await client.eval(ENQUEUE, 1, stream, 1, "first")
        await client.xreadgroup("probe", "consumer", {stream: ">"})
        assert await client.eval(ENQUEUE, 1, stream, 1, "second") is None
        assert (await client.xpending(stream, "probe"))["pending"] == 1
        await client.eval(ACK_DELETE, 1, stream, "probe", first)
        assert await client.xlen(stream) == 0
        assert (await client.xpending(stream, "probe"))["pending"] == 0


async def test_worker_manual_ack_keeps_long_coroutine_alive(redis_url, tmp_path):
    from tests.fixtures.queue_probe import broker, long_probe

    # The worker and sender use the same real Redis, isolated from any developer service.
    broker.connection_pool = BlockingConnectionPool.from_url(redis_url)
    await broker.startup()
    env = dict(os.environ, TEST_PROBE_REDIS_URL=redis_url)
    command = [sys.executable, "-m", "taskiq", "worker",
               "tests.fixtures.queue_probe:broker", "--workers", "1",
               "--max-async-tasks", "1", "--max-prefetch", "0"]
    with (tmp_path / "worker.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=NO_WINDOW)
        try:
            await long_probe.kiq("same-task")
            async with Redis.from_url(redis_url) as client:
                await wait_value(client, "vs:started_count", b"1")
                assert await client.xlen(broker.queue_name) == 0
                assert not await client.exists("vs:finished_count")
                await long_probe.kiq("same-task")
                await long_probe.kiq("other-task")
                await asyncio.sleep(0.2)
                assert await client.get("vs:started_count") == b"1"
                await client.set("vs:release", "yes")
                await wait_value(client, "vs:finished_count", b"2")
                await wait_value(client, "vs:duplicate_count", b"1")
                assert await client.xlen(broker.queue_name) == 0
        finally:
            stop_process(process)
            await broker.shutdown()


async def test_busy_native_worker_does_not_reserve_the_idle_workers_task(redis_url):
    from taskiq import Context, TaskiqDepends
    from vision_shield.jobs import worker
    from vision_shield.jobs.broker import BoundedRedisBroker

    stop, release, first_started, second_started = (asyncio.Event() for _ in range(4))
    seen, listeners = {}, []
    brokers = [BoundedRedisBroker(redis_url, queue_name='vs:sequential-probe',
        consumer_group_name='vs:sequential-probe', consumer_name=name) for name in ('busy', 'idle')]

    async def process(label: str, context: Context = TaskiqDepends()):
        await context.broker.ack_message(context.message.task_id)
        seen[label] = context.broker.consumer_name
        if label == 'first':
            first_started.set()
            await release.wait()
        else:
            second_started.set()

    for broker in brokers:
        broker.state.stop_event = stop
        broker.task(task_name='native.sequential.probe')(process)
        await broker.startup()
    sender = brokers[0].get_all_tasks()['native.sequential.probe']
    try:
        first = worker.Receiver(brokers[0], max_async_tasks=1, max_prefetch=0)
        first.run_startup = False
        listeners.append(asyncio.create_task(first.listen(stop)))
        await sender.kiq('first')
        await asyncio.wait_for(first_started.wait(), 3)
        await sender.kiq('second')
        await asyncio.sleep(.15)
        second = worker.Receiver(brokers[1], max_async_tasks=1, max_prefetch=0)
        second.run_startup = False
        listeners.append(asyncio.create_task(second.listen(stop)))
        await asyncio.wait_for(second_started.wait(), 2)
        assert seen == {'first': 'busy', 'second': 'idle'}
        assert not release.is_set()
    finally:
        stop.set()
        release.set()
        await asyncio.wait_for(asyncio.gather(*listeners), 5)
        for broker in brokers:
            await broker.shutdown()


async def test_stopping_native_worker_leaves_followup_in_redis(redis_url):
    from taskiq import Context, TaskiqDepends
    from vision_shield.jobs import worker
    from vision_shield.jobs.broker import BoundedRedisBroker

    stop, release, started = (asyncio.Event() for _ in range(3))
    seen = []
    broker = BoundedRedisBroker(redis_url, queue_name='vs:native-stop-probe', consumer_group_name='vs:native-stop-probe')
    broker.state.stop_event = stop

    @broker.task
    async def process(label: str, context: Context = TaskiqDepends()):
        await broker.ack_message(context.message.task_id)
        seen.append(label)
        started.set()
        await release.wait()

    await broker.startup()
    receiver = worker.Receiver(broker, max_async_tasks=1, max_prefetch=0)
    receiver.run_startup = False
    listener = asyncio.create_task(receiver.listen(stop))
    try:
        await process.kiq('first')
        await asyncio.wait_for(started.wait(), 3)
        await process.kiq('followup')
        await asyncio.sleep(.15)
        stop.set()
        release.set()
        await asyncio.wait_for(listener, 3)
        assert seen == ['first']
        async with Redis.from_url(redis_url) as client:
            assert await client.xlen(broker.queue_name) == 1
            assert (await client.xpending(broker.queue_name, broker.consumer_group_name))['pending'] == 0
    finally:
        stop.set()
        release.set()
        await asyncio.gather(listener)
        await broker.shutdown()
