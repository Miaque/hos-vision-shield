import asyncio
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile
from pydantic import ValidationError

from vision_shield.api.schemas import (DesiredStateInput, FileUrlInput, StreamInput, TaskResponse, TaskListResponse,
                                      TaskQueryInput, TaskQueryResponse,
                                      ResourceSnapshot, ResourcesResponse)
from vision_shield.jobs.diagnostics import STALE_SECONDS
from vision_shield.jobs.service import require_worker
from vision_shield.jobs.store import IdempotencyConflict, CapacityExceeded, request_fingerprint, stream_health, publisher_exit_pending
from vision_shield.media.source import SourceError, validate_source
from vision_shield.detection.timing import STAGES
from vision_shield.detection.device_profile import CHECKPOINTS

router = APIRouter(prefix="/v1")


def public(request, task, now_ms=None):
    """只暴露公共字段，健康只依据共享存储中的实际输出和租约。"""
    if not request.app.state.settings.debug:
        task = task | {'stats': {key: value for key, value in task['stats'].items() if key != 'stream'}}
    if task["kind"] == "stream" and "generation" in task:
        observed = now_ms if now_ms is not None else task.get("_observed_ms", int(datetime.now(timezone.utc).timestamp() * 1000))
        health = stream_health(task, observed)
        task = task | {"health": health}
    return TaskResponse.from_record(task, request.app.state.settings.rtsp_public_base_url)


def creation_headers(request):
    """在任何外部 I/O 前验证可选幂等键和受限通知目标。"""
    key = request.headers.get("Idempotency-Key")
    target = request.headers.get("X-Notification-Target")
    if key is not None and (not re.fullmatch(r"[\x20-\x7e]{1,128}", key) or not key.strip()):
        raise HTTPException(422, "invalid_input")
    if target is not None and (not key or target not in request.app.state.settings.notification_targets):
        raise HTTPException(422, "invalid_input")
    return key, target


async def replay(request, key, fingerprint):
    """已接纳意图不受当前 worker 或新任务容量影响。"""
    return await request.app.state.store.replay(key, fingerprint,
        notification_target=request.headers.get("X-Notification-Target")) if key else None


@router.get('/diagnostics/resources', response_model=ResourcesResponse)
async def resources(request: Request):
    if not request.app.state.settings.debug:
        raise HTTPException(404, 'diagnostics_disabled')
    client = request.app.state.redis
    seconds, micros = await client.time()
    now = seconds + micros / 1_000_000
    pools = ('files', 'streams')
    groups = await asyncio.gather(*(client.zrangebyscore(f'vs:workers:{pool}',
        int((now - request.app.state.settings.lease_seconds) * 1000), '+inf') for pool in pools))
    workers = [(pool, worker.decode() if isinstance(worker, bytes) else worker)
               for pool, group in zip(pools, groups) for worker in group]
    samples = await client.mget([f'vs:resources:{pool}:{worker}' for pool, worker in workers]) if workers else []
    result = []
    for (pool, worker), raw in zip(workers, samples):
        snapshot, age, reason = None, None, 'no_sample'
        if raw:
            try:
                snapshot = ResourceSnapshot.model_validate_json(raw)
                age, reason = max(0, now - snapshot.sampled_at_unix), None
            except ValidationError:
                reason = 'invalid_sample'
        result.append(dict(worker_id=worker, pool=pool, age_seconds=age,
                           stale=age is None or age > STALE_SECONDS, unavailable_reason=reason, snapshot=snapshot))
    return dict(sampled_at_unix=now, workers=result)


@router.post("/stream-tasks", response_model=TaskResponse, status_code=202)
async def create_stream(data: StreamInput, request: Request, response: Response,
                        detection_interval: Literal['1', '3'] | None = Header(default=None, alias='X-Detection-Interval',
                            description='实时流默认每三帧全量检测；解码、遮挡和输出仍逐帧执行'),
                        tracking: Literal['template', 'flow'] | None = Header(default=None, alias='X-Tracking-Mode',
                            description='默认检测间隔 3 配合 flow；显式间隔 1 时默认 template')):
    key, target = creation_headers(request)
    settings = request.app.state.settings
    interval = int(detection_interval or '3')
    mode = tracking or ('flow' if interval == 3 else 'template')
    if mode == 'flow' and interval != 3:
        raise HTTPException(422, 'invalid_input')
    options = {'detection_interval': interval, 'tracking': mode,
               **({'audio_enabled': False} if not data.audio_enabled else {})}
    fingerprint = request_fingerprint(kind="stream", source_kind="rtsp", source_digest=data.url,
                                      redaction_mode=data.redaction_mode, notification_target=target,
                                      profile_detection=False,
                                      **({"publish_url": data.publish_url} if data.publish_url is not None else {}),
                                      **({"playback_url": data.playback_url} if data.playback_url is not None else {}),
                                      **({"max_bitrate": data.max_bitrate} if data.max_bitrate is not None else {}), **options)
    task = await replay(request, key, fingerprint)
    if task is not None:
        response.headers["Location"] = f"/v1/tasks/{task['id']}"
        return public(request, task)
    await validate_source(data.url, settings.source_allowlist, ("rtsp", "rtmp"))
    if data.publish_url is not None:
        await validate_source(data.publish_url, settings.source_allowlist, ("rtsp", "rtmp"))
    await require_worker(request.app.state.redis, "stream", settings.lease_seconds)
    await require_media_capacity(settings)
    task = await request.app.state.store.create("stream", "rtsp", data.url, redaction_mode=data.redaction_mode,
        idempotency_key=key, fingerprint=fingerprint, notification_target=target, publish_url=data.publish_url,
        playback_url=data.playback_url,
        max_bitrate=data.max_bitrate, **options)
    response.headers["Location"] = f"/v1/tasks/{task['id']}"
    return public(request, task)


@router.post("/file-tasks", response_model=TaskResponse, status_code=202)
async def file_url(data: FileUrlInput, request: Request, response: Response):
    key, target = creation_headers(request)
    settings = request.app.state.settings
    fingerprint = request_fingerprint(kind=data.kind, source_kind="url", source_digest=data.url,
                                      redaction_mode=data.redaction_mode, notification_target=target,
                                      profile_detection=False)
    task = await replay(request, key, fingerprint)
    if task is not None:
        response.headers["Location"] = f"/v1/tasks/{task['id']}"
        return public(request, task)
    await validate_source(data.url, settings.source_allowlist)
    await require_worker(request.app.state.redis, data.kind, settings.lease_seconds)
    await require_media_capacity(settings)
    task = await request.app.state.store.create(data.kind, "url", data.url, redaction_mode=data.redaction_mode,
        idempotency_key=key, fingerprint=fingerprint, notification_target=target)
    response.headers["Location"] = f"/v1/tasks/{task['id']}"
    return public(request, task)


@router.post("/file-tasks/uploads", response_model=TaskResponse, status_code=202,
             openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                 "type": "object", "required": ["kind", "file"], "properties": {
                     "kind": {"type": "string", "enum": ["image", "video"]},
                     "file": {"type": "string", "format": "binary"},
                     "redaction_mode": {"type": "string", "enum": ["solid", "mosaic"], "default": "solid"},
                 }}}}}})
async def upload(request: Request, response: Response,
                 tracking: Literal['template', 'flow'] | None = Header(default=None, alias='X-Tracking-Mode',
                     description='仅离线视频顺序验证：flow 须配合检测间隔 3，跟踪失效立即全图复核'),
                 detection_interval: Literal['1', '3'] | None = Header(default=None, alias='X-Detection-Interval',
                     description='仅离线视频顺序验证：全图检测间隔，省略时保持逐帧；解码、跟踪、遮罩和编码仍逐帧执行'),
                 profile_detection: Literal['1', 'host', 'forward-primary-face', 'forward-primary-plate',
                                            'forward-detail-face', 'forward-detail-plate'] | None = Header(
                     default=None, alias='X-Detection-Profile',
                     description='仅视频：1沿用主车牌全图CUDA采样；host记录调用分布；forward-*按真实前向序号采样指定模型。诊断FPS不作吞吐验收')):
    key, target_id = creation_headers(request)
    settings = request.app.state.settings
    if profile_detection is not None and not settings.debug:
        raise HTTPException(422, 'diagnostics_disabled')
    profile_mode = True if profile_detection == '1' else profile_detection or False
    directory, keep_input, submission_unknown = None, False, False
    try:
        async with asyncio.timeout(settings.upload_timeout):
            async with request.form(max_files=1, max_fields=2) as form:
                if (not {"kind", "file"} <= set(form) or set(form) - {"kind", "file", "redaction_mode"}
                        or len(form.multi_items()) != len(form)):
                    raise HTTPException(422, "invalid_input")
                kind, file = form["kind"], form["file"]
                if kind not in {"image", "video"} or not isinstance(file, UploadFile):
                    raise HTTPException(422, "invalid_input")
                if (profile_detection is not None or detection_interval is not None or tracking is not None) and kind != 'video':
                    raise HTTPException(422, 'invalid_input')
                if tracking == 'flow' and detection_interval != '3':
                    raise HTTPException(422, 'invalid_input')
                redaction_mode = form.get("redaction_mode", "solid")
                if redaction_mode not in ("solid", "mosaic"):
                    raise HTTPException(422, "invalid_input")
                task_id = str(uuid4())
                directory = settings.media_root.resolve() / task_id
                await asyncio.to_thread(directory.mkdir, parents=True)
                source = directory / "source.bin"
                digest = await save_upload(file, source,
                    settings.max_image_bytes if kind == "image" else settings.max_video_bytes)
        fingerprint = request_fingerprint(kind=kind, source_kind="upload", source_digest=digest,
            redaction_mode=redaction_mode, notification_target=target_id, profile_detection=profile_mode,
            **({'detection_interval': 3} if detection_interval == '3' else {}),
            **({'tracking': 'flow'} if tracking == 'flow' else {}))
        task = await replay(request, key, fingerprint)
        if task is None:
            await require_worker(request.app.state.redis, kind, settings.lease_seconds)
            await require_media_capacity(settings)
            # Redis 超时可能已经提交，保留输入供同键恢复，不能盲删。
            submission_unknown = True
            task = await request.app.state.store.create(kind, "upload", str(source), task_id,
                redaction_mode=redaction_mode, idempotency_key=key, fingerprint=fingerprint,
                notification_target=target_id,
                **({'tracking': tracking} if tracking is not None else {}),
                **({'detection_interval': int(detection_interval)} if detection_interval is not None else {}),
                **({'profile_detection': profile_mode} if profile_detection is not None else {}))
            submission_unknown = False
        keep_input = task["id"] == task_id
        response.headers["Location"] = f"/v1/tasks/{task['id']}"
        return public(request, task)
    except (IdempotencyConflict, CapacityExceeded):
        submission_unknown = False
        raise
    finally:
        if directory is not None and not keep_input and not submission_unknown:
            await asyncio.to_thread(shutil.rmtree, directory)


async def save_upload(file, source, limit):
    """有界写入并计算真实文件摘要，失败也确保关闭句柄。"""
    output = await asyncio.to_thread(source.open, "wb")
    digest, size = hashlib.sha256(), 0
    try:
        while block := await file.read(65536):
            size += len(block)
            if size > limit:
                raise SourceError("media_too_large", 413)
            await asyncio.to_thread(output.write, block)
            digest.update(block)
    finally:
        await asyncio.to_thread(output.close)
    return digest.hexdigest()


async def require_media_capacity(settings):
    """磁盘高水位拒绝新接纳，旧幂等意图仍可恢复，不自动删结果。"""
    def available():
        root = settings.media_root.resolve()
        while not root.exists():
            root = root.parent
        usage = shutil.disk_usage(root)
        return usage.used / usage.total < settings.media_disk_high_watermark
    if not await asyncio.to_thread(available):
        raise CapacityExceeded


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(request: Request):
    groups = {"current": [], "queued": []}
    tasks = await request.app.state.store.active()
    for task in sorted(tasks, key=lambda task: (task["created_at"], task["id"])):
        if task["state"] not in {"queued", "processing", "connecting", "running", "recovering"}:
            continue
        waiting = task["desired_state"] == "run" and not publisher_exit_pending(task) and (
            task["state"] == "queued" or (task["state"] == "recovering" and task["owner_id"] is None)
        )
        groups["queued" if waiting else "current"].append(public(request, task))
    return groups


@router.post("/tasks/query", response_model=TaskQueryResponse)
async def query_tasks(data: TaskQueryInput, request: Request):
    """只查询给定的至多一百项，支持已退出活跃集合的终态。"""
    ids = list(dict.fromkeys(str(value) for value in data.task_ids))
    records = await request.app.state.store.query(ids)
    seconds, micros = await request.app.state.redis.time()
    items = [public(request, task, seconds * 1000 + micros // 1000) for task in records]
    found = {task.id for task in items}
    return dict(items=items, missing_ids=[value for value in ids if value not in found])


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: UUID, request: Request, response: Response):
    task = await request.app.state.store.get(str(task_id))
    if task is None:
        raise HTTPException(404, "task_not_found")
    if request.app.state.settings.debug and task.get("video_encoder") in {"libx264", "h264_nvenc"}:
        response.headers["X-Video-Encoder"] = task["video_encoder"]
    if request.app.state.settings.debug and task.get("video_decoder") in {"h264", "h264_nvdec"}:
        response.headers["X-Video-Decoder"] = task["video_decoder"]
    if request.app.state.settings.debug and task.get('detection_stats'):
        response.headers['X-Detection-Counts'] = json.dumps(task['detection_stats'], separators=(',', ':'))
    if request.app.state.settings.debug and task.get("timings"):
        response.headers["Server-Timing"] = ", ".join(
            f"task-{name};dur={task['timings'][name] * 1000:.3f}"
            for name in ("frame", "inference", "encode", "candidate-primary", "candidate-detail", *STAGES)
            if name in task["timings"])
    return public(request, task)


@router.get('/tasks/{task_id}/diagnostics')
async def detection_diagnostics(task_id: UUID, request: Request,
                                sample: int | None = Query(default=None, json_schema_extra={'enum': list(CHECKPOINTS)})):
    if not request.app.state.settings.debug:
        raise HTTPException(404, 'diagnostics_disabled')
    if sample is not None and sample not in CHECKPOINTS:
        raise HTTPException(422, 'invalid_input')
    task = await request.app.state.store.get(str(task_id))
    if task is None:
        raise HTTPException(404, 'task_not_found')
    if task.get('kind') == 'stream' and sample is None and (task.get('media_failure') or task.get('stream_attempts')):
        result = {'mode': 'stream'}
        if task.get('media_failure'):
            result['last_failure'] = task['media_failure']
        if task.get('stream_attempts'):
            result.update(attempts=task['stream_attempts'], attempts_discarded=task.get('stream_attempts_discarded', 0))
        return result
    if task.get('device_profile') is None:
        raise HTTPException(409, 'diagnostics_not_ready')
    if sample is None:
        return task['device_profile']
    root = request.app.state.settings.media_root.resolve()
    mode = task['device_profile'].get('mode')
    path = (f'host-profile/{mode}-{sample}.json' if mode in (
        'forward-primary-face', 'forward-primary-plate', 'forward-detail-face', 'forward-detail-plate')
        else f'plate-profile-{sample}.json')
    target = await asyncio.to_thread((root / str(task_id) / f"attempt-{task['generation']}" / path).resolve)
    if not target.is_relative_to(root / str(task_id)) or not await asyncio.to_thread(target.is_file):
        raise HTTPException(409, 'diagnostics_not_ready')
    return FileResponse(target, media_type='application/json', filename=target.name)


@router.patch("/tasks/{task_id}", response_model=TaskResponse, status_code=202)
async def stop_task(task_id: UUID, data: DesiredStateInput, request: Request, response: Response):
    task = await request.app.state.store.request_stop(str(task_id), data.desired_state)
    if task is None:
        raise HTTPException(404, "task_not_found")
    if task["state"] == data.desired_state:
        response.status_code = 200
    return public(request, task)


@router.get("/tasks/{task_id}/result")
async def result(task_id: UUID, request: Request):
    task = await request.app.state.store.get(str(task_id))
    if task is None:
        raise HTTPException(404, "task_not_found")
    if task["state"] != "succeeded" or task["kind"] == "stream":
        raise HTTPException(409, "result_not_ready")
    root = request.app.state.settings.media_root.resolve()
    target = await asyncio.to_thread((root / task["result_ref"]).resolve)
    if not target.is_relative_to(root / str(task_id)) or not await asyncio.to_thread(target.is_file):
        raise HTTPException(409, "result_not_ready")
    return FileResponse(target, filename=target.name)
