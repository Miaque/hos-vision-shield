"""Native single-process Taskiq receiver; start more processes to add slots."""

import argparse
import asyncio
import contextlib
import logging
from uuid import uuid4

from redis.asyncio import Redis
from taskiq.receiver import Receiver

from vision_shield.config import Settings
from vision_shield.jobs.tasks import file_broker, stream_broker


async def main(pool):
    settings = Settings()
    if not all([await asyncio.to_thread(path.is_file) for path in (settings.face_model_path, settings.plate_model_path)]):
        raise RuntimeError("Both official model files are required")
    broker = file_broker if pool == "files" else stream_broker
    broker.is_worker_process = True
    worker_id = str(uuid4())
    client = Redis(connection_pool=broker.connection_pool)

    async def announce():
        while True:
            seconds, micros = await client.time()
            now = seconds * 1000 + micros // 1000
            await client.zadd(f"vs:workers:{pool}", {worker_id: now})
            await asyncio.sleep(settings.heartbeat_seconds)

    receiver = Receiver(broker, max_async_tasks=1, max_prefetch=0, wait_tasks_timeout=settings.process_stop_timeout)
    heartbeat = None
    try:
        await broker.startup()
        heartbeat = asyncio.create_task(announce())
        receiver.run_startup = False
        await receiver.listen(asyncio.Event())
    finally:
        if heartbeat:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        await client.zrem(f"vs:workers:{pool}", worker_id)
        await broker.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", choices=("files", "streams"))
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(parser.parse_args().pool))
