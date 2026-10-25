import asyncio
import contextlib
import logging
from pathlib import Path
import sys
import time
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
from vision_shield.media.attempts import failure_details


def log_model_timing(event: dict, task_id: str, generation: int) -> None:
    """模型冷加载单独记日志，不混入既有媒体处理 stats。"""
    if "model_load_seconds" in event:
        logging.getLogger(__name__).info("task=%s generation=%s 模型%s耗时=%.3fs",
                                         task_id, generation, '复用' if event.get('model_reused') else '加载',
                                         event["model_load_seconds"])


async def execute(task_id, broker, context):
    deployment_stop = getattr(broker.state, "stop_event", None)
    if deployment_stop is not None and deployment_stop.is_set():
        return
    settings = Settings()
    session = getattr(broker.state, 'media_process', None)
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
    cancellation = {}

    async def control():
        try:
            while True:
                async with asyncio.timeout(settings.lease_seconds / 2):
                    current = await store.get(task_id)
                    if not current:
                        cancellation.update(reason="task_removed", phase="worker_control")
                        cancel.set()
                        return
                    if current["desired_state"] != "run":
                        cancellation.update(reason="stop_requested", phase="worker_control")
                        # Hold ownership until the child has actually stopped.
                        cancel.set()
                    if deployment_stop is not None and deployment_stop.is_set():
                        cancellation.update(reason="worker_shutdown", phase="worker_control")
                        # 部署停止期间仍续租，直到媒体子进程完成回收。
                        cancel.set()
                    if not await store.heartbeat(task_id, owner, generation):
                        cancellation.update(reason="lease_lost", phase="worker_control")
                        cancel.set()
                        return
                await asyncio.sleep(min(settings.heartbeat_seconds, 1))
        except Exception as error:
            cancellation.update(failure_details(error, "worker_control"))
            cancel.set()

    controller = asyncio.create_task(control())
    try:
        if record["kind"] == "stream":
            await run_stream(record, store, settings, cancel, owner, generation,
                             cancellation=cancellation, session=session)
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
            if "stats" in event or "device_profile" in event or "detection_stats" in event:
                await store.update_owned(task_id, owner, generation,
                                         **{key: event[key] for key in ("stats", "timings", "video_encoder", "video_decoder", "device_profile", "detection_stats") if key in event})

        result = {}
        if not cancel.is_set():
            payload = dict(kind=record["kind"], source=str(source), directory=str(directory),
                           config={**settings.model_dump(mode="json"), "redaction_mode": record.get("redaction_mode", "solid")})
            if record.get('profile_detection'):
                payload['profile_detection'] = record['profile_detection']
            if record.get('detection_interval'):
                payload['config']['detection_interval'] = record['detection_interval']
            if record.get('tracking'):
                payload['config']['tracking'] = record['tracking']
            result = await run_process([sys.executable, "-u", "-m", "vision_shield.media.child"],
                                       payload, cancel, event_received, settings.process_stop_timeout,
                                       **({'session': session} if session is not None else {}))
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


async def run_stream(record, store, settings, cancel, owner, generation, *, cancellation=None, session=None):
    task_id, retry = record["id"], 0
    publisher = Publisher(store, settings)
    history = list(record.get("stream_attempts", []))
    discarded = record.get("stream_attempts_discarded", 0)
    for previous in history:
        if "ended_at_unix" not in previous:
            previous.update(reason="worker_lost", phase="worker_control", ended_at_unix=time.time())

    async def persist_ending():
        attempt.update(ending, ended_at_unix=time.time())
        await store.update_owned(task_id, owner, generation, stream_attempts=history,
                                 stream_attempts_discarded=discarded)

    async def media_closed():
        nonlocal phase
        if cancel.is_set():
            ending.update(cancellation or dict(reason="control_cancelled", phase="worker_control"))
        elif not ending:
            ending.update(reason="child_exit", phase="child_process")
        # External teardown may finalize the task and release ownership in this callback.
        await persist_ending()
        try:
            await publisher.close(record, owner, generation,
                                  **({'media_closed': True} if session is not None else {'process_exited': True}))
        except Exception:
            phase = "publisher_close"
            raise

    try:
        while not cancel.is_set():
            attempt = dict(run_id=uuid4().hex, generation=generation, started_at_unix=time.time(),
                           started_monotonic=time.monotonic())
            history.append(attempt)
            discarded += max(0, len(history) - 32)
            history = history[-32:]
            await store.update_owned(task_id, owner, generation, stream_attempts=history,
                                     stream_attempts_discarded=discarded)
            ending, phase, finish_error = {}, "source_validate", None
            try:
                original, address = await validate_source(record["source_ref"], settings.source_allowlist, ("rtsp",))
                phase = "publisher_prepare"
                publish_url = await publisher.prepare(record, owner, generation)
                await store.update_owned(task_id, owner, generation, state="connecting", ready=False)

                async def event_received(event):
                    log_model_timing(event, task_id, generation)
                    if event.get("failure_type"):
                        logging.getLogger(__name__).warning("task=%s media failure type=%s at=%s", task_id, event["failure_type"], event.get("failure_at"))
                    values = {}
                    if event.get("stream_end"):
                        ending.update(event["stream_end"])
                    stream = event.get("stats", {}).get("stream")
                    if stream and "attempt_id" not in attempt:
                        attempt["attempt_id"] = stream["attempt_id"]
                        values["stream_attempts"] = history
                    if event.get("failure_type"):
                        values["media_failure"] = dict(type=event["failure_type"], at=event.get("failure_at", []),
                                                       generation=generation, observed_at_unix=time.time())
                    if event.get("output"):
                        # 父子进程共享单调时钟，管道积压不能把陈旧输出改成新鲜证明。
                        age_ms = max(0, int((time.monotonic() - event["output_monotonic"]) * 1000))
                        confirmed = await store.mark_output(task_id, owner, generation, age_ms=age_ms)
                    else:
                        confirmed = False
                    if event.get("ready") and confirmed:
                        values.update(state="running", ready=True, error=None)
                    values.update({key: event[key] for key in ("stats", "timings", "video_encoder", "video_decoder", "detection_stats")
                                   if key in event})
                    if values:
                        await store.update_owned(task_id, owner, generation, **values)

                payload = dict(kind="stream", source=str(original.copy_with(host=address)),
                               publish_url=publish_url, config={**settings.model_dump(mode="json"), "redaction_mode": record.get("redaction_mode", "solid"),
                                                               "publish_url": record.get("publish_url"),
                                                               "stream_started_monotonic": attempt['started_monotonic']},
                               directory=str(settings.media_root.resolve() / task_id / f"attempt-{generation}"))
                for key in ('detection_interval', 'tracking', 'max_bitrate'):
                    if key in record:
                        payload['config'][key] = record[key]
                phase = "child_process"
                result = await run_process([sys.executable, "-u", "-m", "vision_shield.media.child"],
                                           payload, cancel, event_received, settings.process_stop_timeout,
                                            on_exit=media_closed if record.get("publish_url") else None,
                                            **({'session': session} if session is not None else {}))
                if not ending:
                    ending.update(reason="child_exit", phase=phase, returncode=result.get("returncode"))
                ending["returncode"] = result.get("returncode")
                if result.get("failure_type") and ending["reason"] == "child_exit":
                    ending.update(reason="child_initialize_error", phase="child_initialize",
                                  error_type=result["failure_type"], at=result.get("failure_at", []))
                    if "failure_errno" in result:
                        ending["errno"] = result["failure_errno"]
                if cancel.is_set():
                    ending.update(cancellation or dict(reason="control_cancelled", phase="worker_control"))
                    break
                phase = "publisher_close"
                await publisher.close(record, owner, generation)
                if result.get("error") and not result.get("retryable"):
                    finish_error = result["error"]
            except SourceError as error:
                ending.update(failure_details(error, phase))
                if error.code != "source_unreachable":
                    raise
            except (httpx.HTTPError, TimeoutError) as error:
                ending.update(failure_details(error, phase))
            except Exception as error:
                ending.update(failure_details(error, phase))
                raise
            finally:
                if not ending:
                    ending.update(cancellation or dict(reason="worker_interrupted", phase=phase))
                await persist_ending()
                logging.getLogger(__name__).info("task=%s attempt=%s reason=%s phase=%s type=%s errno=%s",
                    task_id, attempt.get("attempt_id", attempt["run_id"]), attempt.get("reason"),
                    attempt.get("phase"), attempt.get("error_type"), attempt.get("errno"))
            if finish_error:
                await store.finish(task_id, owner, generation, "failed", error=finish_error)
                return
            await store.update_owned(task_id, owner, generation, state="recovering", ready=False,
                                     error={"code": "source_unreachable" if ending.get("phase") in (
                                         "source_validate", "source_open", "source_read") else "processing_failed",
                                            "message": "Stream reconnect pending"})
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
