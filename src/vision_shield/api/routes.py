import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile
from pydantic import ValidationError

from vision_shield.api.schemas import (DesiredStateInput, FileUrlInput, StreamInput, TaskResponse, TaskListResponse,
                                      ResourceSnapshot, ResourcesResponse)
from vision_shield.jobs.diagnostics import STALE_SECONDS
from vision_shield.jobs.service import require_worker
from vision_shield.media.source import SourceError, validate_source

router = APIRouter(prefix="/v1")


def public(request, task):
    return TaskResponse.from_record(task, request.app.state.settings.rtsp_public_base_url)


@router.get('/diagnostics/resources', response_model=ResourcesResponse)
async def resources(request: Request):
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
async def create_stream(data: StreamInput, request: Request, response: Response):
    settings = request.app.state.settings
    await validate_source(data.url, settings.source_allowlist, ("rtsp",))
    await require_worker(request.app.state.redis, "stream", settings.lease_seconds)
    task = await request.app.state.store.create("stream", "rtsp", data.url)
    response.headers["Location"] = f"/v1/tasks/{task['id']}"
    return public(request, task)


@router.post("/file-tasks", response_model=TaskResponse, status_code=202)
async def file_url(data: FileUrlInput, request: Request, response: Response):
    settings = request.app.state.settings
    await validate_source(data.url, settings.source_allowlist)
    await require_worker(request.app.state.redis, data.kind, settings.lease_seconds)
    task = await request.app.state.store.create(data.kind, "url", data.url)
    response.headers["Location"] = f"/v1/tasks/{task['id']}"
    return public(request, task)


@router.post("/file-tasks/uploads", response_model=TaskResponse, status_code=202,
             openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                 "type": "object", "required": ["kind", "file"], "properties": {
                     "kind": {"type": "string", "enum": ["image", "video"]},
                     "file": {"type": "string", "format": "binary"},
                 }}}}}})
async def upload(request: Request, response: Response):
    settings = request.app.state.settings
    await require_worker(request.app.state.redis, "image", settings.lease_seconds)
    async with asyncio.timeout(settings.upload_timeout):
        async with request.form(max_files=1, max_fields=1) as form:
            if set(form) != {"kind", "file"} or len(form.multi_items()) != 2:
                raise HTTPException(422, "invalid_input")
            kind, file = form["kind"], form["file"]
            if kind not in {"image", "video"} or not isinstance(file, UploadFile):
                raise HTTPException(422, "invalid_input")
            task_id = str(uuid4())
            directory = settings.media_root.resolve() / task_id
            await asyncio.to_thread(directory.mkdir, parents=True)
            target = directory / "source.bin"
            output = await asyncio.to_thread(target.open, "wb")
            size = 0
            try:
                while block := await file.read(65536):
                    size += len(block)
                    if size > (settings.max_image_bytes if kind == "image" else settings.max_video_bytes):
                        raise SourceError("media_too_large", 413)
                    await asyncio.to_thread(output.write, block)
            finally:
                await asyncio.to_thread(output.close)
    task = await request.app.state.store.create(kind, "upload", str(target), task_id)
    response.headers["Location"] = f"/v1/tasks/{task['id']}"
    return public(request, task)


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(request: Request):
    groups = {"current": [], "queued": []}
    tasks = await request.app.state.store.active()
    for task in sorted(tasks, key=lambda task: (task["created_at"], task["id"])):
        if task["state"] not in {"queued", "processing", "connecting", "running", "recovering"}:
            continue
        waiting = task["desired_state"] == "run" and (
            task["state"] == "queued" or (task["state"] == "recovering" and task["owner_id"] is None)
        )
        groups["queued" if waiting else "current"].append(public(request, task))
    return groups


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: UUID, request: Request):
    task = await request.app.state.store.get(str(task_id))
    if task is None:
        raise HTTPException(404, "task_not_found")
    return public(request, task)


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
