"""Native single-process Taskiq receiver; start more processes to add slots."""

import argparse
import asyncio
import contextlib
import logging
import json
import os
import signal
import sys
from uuid import uuid4

from redis.asyncio import Redis
from taskiq.receiver import Receiver

from vision_shield.config import Settings
from vision_shield.jobs.tasks import file_broker, stream_broker
from vision_shield.media.runner import run_process

PROBE_TIMEOUT = 120


async def probe_models(settings: Settings, stop: asyncio.Event) -> None:
    """预检受现有子进程监督，成功事件和正常退出缺一不可。"""
    async def event_received(event):
        pass
    try:
        async with asyncio.timeout(PROBE_TIMEOUT):
            result = await run_process([sys.executable, "-u", "-m", "vision_shield.media.child"],
                                       dict(kind="probe", config=settings.model_dump(mode="json")),
                                       stop, event_received, settings.process_stop_timeout)
    except TimeoutError:
        raise RuntimeError("模型预检超时，子进程已回收") from None
    if "probe" not in result or result.get("returncode") != 0 or result.get("cancelled") or stop.is_set():
        raise RuntimeError(result.get("preflight_error") or "模型预检失败或已停止")
    logging.getLogger(__name__).info("模型预检通过：%s", json.dumps(result["probe"], ensure_ascii=False))


async def main(pool, stop=None):
    settings = Settings()
    stop = stop if stop is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    if os.name != "nt":
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, stop.set)
    worker_id = str(uuid4())
    client = None
    broker = None

    async def announce():
        while True:
            seconds, micros = await client.time()
            now = seconds * 1000 + micros // 1000
            await client.zadd(f"vs:workers:{pool}", {worker_id: now})
            await asyncio.sleep(settings.heartbeat_seconds)

    heartbeat = None
    try:
        await probe_models(settings, stop)
        if stop.is_set():
            return
        broker = file_broker if pool == "files" else stream_broker
        broker.state.stop_event = stop
        broker.is_worker_process = True
        client = Redis(connection_pool=broker.connection_pool)
        await broker.startup()
        receiver = Receiver(broker, max_async_tasks=1, max_prefetch=0, wait_tasks_timeout=settings.process_stop_timeout)
        heartbeat = asyncio.create_task(announce())
        receiver.run_startup = False
        await receiver.listen(stop)
    finally:
        stop.set()
        if heartbeat:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            if client:
                await client.zrem(f"vs:workers:{pool}", worker_id)
        finally:
            if broker:
                await broker.shutdown()
            if client:
                await client.aclose()
            if os.name != "nt":
                for signum in (signal.SIGTERM, signal.SIGINT):
                    loop.remove_signal_handler(signum)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", choices=("files", "streams"))
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(parser.parse_args().pool))
