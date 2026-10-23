import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException

from vision_shield.config import Settings
from vision_shield.jobs.store import CapacityExceeded, IdempotencyConflict, InvalidTransition, TaskStore
from vision_shield.media.source import SourceError
from vision_shield.api.routes import router
from vision_shield.api.internal import router as internal_router
from vision_shield.jobs.service import require_worker
import httpx


class BodyLimit:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.settings.max_video_bytes + 65536 if scope["path"].endswith("/uploads") else 65536
        size = 0

        async def bounded_receive():
            nonlocal size
            try:
                message = await asyncio.wait_for(receive(), self.settings.http_read_timeout)
            except TimeoutError:
                raise HTTPException(408, "request_timeout")
            size += len(message.get("body", b""))
            if size > limit:
                raise HTTPException(413, "media_too_large")
            return message

        await self.app(scope, bounded_receive, send)


def error_response(code, status, message=None, headers=None):
    return JSONResponse({"error": {"code": code, "message": message or code.replace("_", " ")}},
                        status_code=status, headers=headers)


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        client = Redis.from_url(settings.redis_url.get_secret_value(), socket_connect_timeout=2, socket_timeout=2)
        app.state.redis = client
        app.state.settings = settings
        app.state.store = TaskStore(client, settings.file_task_limit, settings.stream_task_limit, settings.lease_seconds,
                                    notification_limit=settings.notification_limit)
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="Vision Shield", lifespan=lifespan)
    app.include_router(router)
    app.include_router(internal_router)
    app.add_middleware(BodyLimit, settings=settings)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        return error_response("invalid_input", 422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(str(exc.detail), exc.status_code, headers=exc.headers)

    @app.exception_handler(SourceError)
    async def source_error(request, exc):
        return error_response(exc.code, exc.status)

    @app.exception_handler(CapacityExceeded)
    async def capacity_error(request, exc):
        return error_response("capacity_exceeded", 429, headers={"Retry-After": "5"})

    @app.exception_handler(TimeoutError)
    async def request_timeout(request, exc):
        return error_response("request_timeout", 408)

    @app.exception_handler(InvalidTransition)
    async def transition_error(request, exc):
        return error_response("invalid_transition", 409)

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_error(request, exc):
        """相同提交键不能更换输入内容。"""
        return error_response("idempotency_conflict", 409)

    @app.exception_handler(RedisError)
    async def dependency_error(request, exc):
        return error_response("dependency_unavailable", 503)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        return error_response("internal_error", 500)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request):
        await request.app.state.redis.ping()
        await require_worker(request.app.state.redis, "image", settings.lease_seconds)
        await require_worker(request.app.state.redis, "stream", settings.lease_seconds)
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(f"{settings.mediamtx_control_url}/v3/paths/list")
                response.raise_for_status()
        except httpx.HTTPError:
            raise HTTPException(503, "dependency_unavailable")
        return {"status": "ready"}

    return app


app = create_app()
