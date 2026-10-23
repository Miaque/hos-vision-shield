"""Recover durable intent independently of Taskiq result retention."""

import asyncio
import contextlib
import logging
import os
import signal
from uuid import uuid4

from redis.asyncio import Redis

from vision_shield.config import Settings
from vision_shield.jobs.store import TaskStore, publisher_exit_pending


async def reconcile_once(store, dispatch, stop_stream=None):
    await store.recover_expired()
    for task in await store.active():
        if task["kind"] == "stream" and task["desired_state"] == "stopped":
            if stop_stream:
                await stop_stream(task)
        elif task["state"] in {"queued", "recovering"} and task["desired_state"] == "run":
            if publisher_exit_pending(task):
                continue
            await dispatch(task)


async def main():
    from vision_shield.jobs.broker import file_broker, stream_broker
    from vision_shield.jobs.tasks import process_file, process_stream
    from vision_shield.media.publisher import Publisher

    settings = Settings()
    current = asyncio.current_task()
    if os.name != "nt":
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, current.cancel)
    owner = str(uuid4())
    client = Redis.from_url(settings.redis_url.get_secret_value(), socket_connect_timeout=2, socket_timeout=2)
    store = TaskStore(client, settings.file_task_limit, settings.stream_task_limit, settings.lease_seconds)

    async def dispatch(task):
        function = process_stream if task["kind"] == "stream" else process_file
        await function.kicker().with_labels(job_id=task["id"]).kiq(task["id"])

    try:
        await file_broker.startup()
        await stream_broker.startup()
        while True:
            try:
                leader = await client.eval(
                    "if redis.call('GET',KEYS[1])==ARGV[1] then redis.call('PEXPIRE',KEYS[1],20000); return 1 end; return redis.call('SET',KEYS[1],ARGV[1],'NX','PX',20000)",
                    1, "vs:reconciler", owner,
                )
                if leader:
                    await reconcile_once(store, dispatch, Publisher(store, settings).close)
            except Exception:
                logging.getLogger(__name__).error("Reconciliation failed; durable intent retained")
            await asyncio.sleep(settings.reconcile_seconds)
    finally:
        await file_broker.shutdown()
        await stream_broker.shutdown()
        await client.aclose()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(main())
