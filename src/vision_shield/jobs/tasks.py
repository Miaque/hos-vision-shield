import asyncio
import contextlib
import logging
from pathlib import Path
import sys
from uuid import uuid4

import httpx
from redis.asyncio import Redis
from taskiq import Context, TaskiqDepends

from vision_shield.config import Settings
from vision_shield.jobs.broker import file_broker, stream_broker
from vision_shield.jobs.store import TaskStore
from vision_shield.media.runner import run_process
from vision_shield.media.source import SourceError, download, validate_source
from vision_shield.media.publisher import Publisher


def log_model_timing(event: dict, task_id: str, generation: int) -> None:
    """模型冷加载单独记日志，不混入既有媒体处理 stats。"""
    if "model_load_seconds" in event:
        logging.getLogger(__name__).info("task=%s generation=%s 模型加载耗时=%.3fs",
                                         task_id, generation, event["model_load_seconds"])


async def execute(task_id, broker, context):
    deployment_stop = getattr(broker.state, "stop_event", None)
    if deployment_stop is not None and deployment_stop.is_set():
        return
    settings = Settings()
    client = Redis(connection_pool=broker.connection_pool)
    store = TaskStore(client, settings.file_task_limit, settings.stream_task_limit, settings.lease_seconds)
    owner = str(uuid4())
    record = await store.claim(task_id, owner)
    await broker.ack_message(context.message.task_id)
    if not record:
        return
    generation = record["generation"]
    logger = logging.getLogger(__name__)
    logger.info("task=%s kind=%s generation=%s started", task_id, record["kind"], generation)
    cancel = asyncio.Event()

    async def control():
        try:
            while True:
                async with asyncio.timeout(settings.lease_seconds / 2):
                    current = await store.get(task_id)
                    if not current:
                        cancel.set()
                        return
                    if current["desired_state"] != "run":
                        # Hold ownership until the child has actually stopped.
                        cancel.set()
                    if deployment_stop is not None and deployment_stop.is_set():
                        # 部署停止期间仍续租，直到媒体子进程完成回收。
                        cancel.set()
                    if not await store.heartbeat(task_id, owner, generation):
                        cancel.set()
                        return
                await asyncio.sleep(min(settings.heartbeat_seconds, 1))
        except Exception:
            cancel.set()

    controller = asyncio.create_task(control())
    try:
        if record["kind"] == "stream":
            await run_stream(record, store, settings, cancel, owner, generation)
            return
        root = settings.media_root.resolve()
        directory = root / task_id / f"attempt-{generation}"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        source = record["source_ref"]
        if record["source_kind"] == "url":
            source = root / task_id / "source.bin"
            downloading = asyncio.create_task(download(record["source_ref"], source, settings, record["kind"]))
            stopping = asyncio.create_task(cancel.wait())
            try:
                done, _ = await asyncio.wait({downloading, stopping}, return_when=asyncio.FIRST_COMPLETED)
                if stopping in done:
                    downloading.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await downloading
                else:
                    await downloading
            finally:
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)

        async def event_received(event):
            log_model_timing(event, task_id, generation)
            if event.get("failure_type"):
                logger.warning("task=%s media failure type=%s", task_id, event["failure_type"])
            if "stats" in event or "device_profile" in event:
                await store.update_owned(task_id, owner, generation,
                                         **{key: event[key] for key in ("stats", "timings", "video_encoder", "video_decoder", "device_profile") if key in event})

        result = {}
        if not cancel.is_set():
            payload = dict(kind=record["kind"], source=str(source), directory=str(directory),
                           config={**settings.model_dump(mode="json"), "redaction_mode": record.get("redaction_mode", "solid")})
            if record.get('profile_detection'):
                payload['profile_detection'] = True
            result = await run_process([sys.executable, "-u", "-m", "vision_shield.media.child"],
                                       payload, cancel, event_received, settings.process_stop_timeout)
        current = await store.get(task_id)
        if deployment_stop is not None and deployment_stop.is_set() and current and current["desired_state"] == "run":
            # 未完成文件交给失租约恢复，不将部署停机写成处理失败。
            return
        if current["desired_state"] != "run":
            await store.finish(task_id, owner, generation, current["desired_state"])
        elif result.get("returncode") == 0 and result.get("result_ref"):
            relative = str(Path(result["result_ref"]).relative_to(root))
            await store.finish(task_id, owner, generation, "succeeded", relative)
        else:
            await store.finish(task_id, owner, generation, "failed", error=result.get("error") or
                               {"code": "processing_failed", "message": "Media processing failed"})
    except Exception as error:
        logger.warning("task=%s execution failure type=%s", task_id, type(error).__name__)
        if deployment_stop is not None and deployment_stop.is_set():
            return
        current = await store.get(task_id)
        if record["kind"] == "stream" and current and current["desired_state"] == "stopped":
            # An unreachable control API cannot be treated as confirmed teardown.
            return
        state = current["desired_state"] if current and current["desired_state"] != "run" else "failed"
        code = error.code if isinstance(error, SourceError) else "storage_full" if isinstance(error, OSError) and error.errno == 28 else "processing_failed"
        await store.finish(task_id, owner, generation, state, error={"code": code, "message": code.replace("_", " ")})
    finally:
        controller.cancel()
        await asyncio.gather(controller, return_exceptions=True)
        logger.info("task=%s generation=%s execution ended", task_id, generation)


async def run_stream(record, store, settings, cancel, owner, generation):
    task_id, retry = record["id"], 0
    publisher = Publisher(store, settings)
    try:
        while not cancel.is_set():
            try:
                original, address = await validate_source(record["source_ref"], settings.source_allowlist, ("rtsp",))
                publish_url = await publisher.prepare(record, owner, generation)
                await store.update_owned(task_id, owner, generation, state="connecting", ready=False)

                async def event_received(event):
                    log_model_timing(event, task_id, generation)
                    if event.get("failure_type"):
                        logging.getLogger(__name__).warning("task=%s media failure type=%s at=%s", task_id, event["failure_type"], event.get("failure_at"))
                    values = {}
                    if event.get("ready"):
                        values.update(state="running", ready=True, error=None)
                    if "stats" in event:
                        values["stats"] = event["stats"]
                    if "timings" in event:
                        values["timings"] = event["timings"]
                    if values:
                        await store.update_owned(task_id, owner, generation, **values)

                payload = dict(kind="stream", source=str(original.copy_with(host=address)),
                               publish_url=publish_url, config={**settings.model_dump(mode="json"), "redaction_mode": record.get("redaction_mode", "solid")},
                               directory=str(settings.media_root.resolve() / task_id / f"attempt-{generation}"))
                result = await run_process([sys.executable, "-u", "-m", "vision_shield.media.child"],
                                           payload, cancel, event_received, settings.process_stop_timeout)
                if cancel.is_set():
                    break
                await publisher.close(record, owner, generation)
                if result.get("error") and not result.get("retryable"):
                    await store.finish(task_id, owner, generation, "failed", error=result["error"])
                    return
            except SourceError as error:
                if error.code != "source_unreachable":
                    raise
            except (httpx.HTTPError, TimeoutError):
                pass
            await store.update_owned(task_id, owner, generation, state="recovering", ready=False,
                                     error={"code": "source_unreachable", "message": "Stream reconnect pending"})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(cancel.wait(), min(30, 2 ** min(retry, 5)))
            retry += 1
    finally:
        await publisher.close(record, owner, generation)
    current = await store.get(task_id)
    if current and current["desired_state"] == "stopped":
        await store.finish(task_id, owner, generation, "stopped")


@file_broker.task
async def process_file(task_id: str, context: Context = TaskiqDepends()):
    await execute(task_id, file_broker, context)


@stream_broker.task
async def process_stream(task_id: str, context: Context = TaskiqDepends()):
    await execute(task_id, stream_broker, context)
