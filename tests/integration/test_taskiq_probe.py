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
