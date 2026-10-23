"""内部推流地址不得改变调用方的播放地址。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from vision_shield.config import Settings
from vision_shield.media.publisher import Publisher
from vision_shield.api.routes import public
from vision_shield.media.source import SourceError


@pytest.mark.parametrize("address", ["http://media:8554", "rtsp://u:p@media", "rtsp://media?q=1", "rtsp://media#x"])
def test_internal_publish_address_reuses_service_validation(address):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rtsp_publish_base_url=address)


@pytest.mark.parametrize("internal,expected", [(None, "external"), ("rtsp://mediamtx:8554/", "mediamtx")])
async def test_publisher_uses_internal_address_and_response_stays_public(internal, expected):
    settings = Settings(_env_file=None, rtsp_public_base_url="rtsp://external:8554",
                        rtsp_publish_base_url=internal)
    publisher = Publisher(None, settings)
    publisher.close = AsyncMock(return_value=True)
    publisher._change = AsyncMock(return_value=True)
    record = dict(id="job", kind="stream", state="running", desired_state="run", ready=True,
                  created_at="2026-10-22T00:00:00Z", updated_at="2026-10-22T00:00:00Z", error=None, stats={})
    address = urlsplit(await publisher.prepare(record, "owner", 2))
    assert address.hostname == expected and address.username == "job.2"
    assert address.password == "owner"
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    assert public(request, record).playback_url == "rtsp://external:8554/redacted/job"


@pytest.mark.parametrize("protocol,path,allowed", [("rtsp", "redacted/job", True), ("hls", "redacted/job", False), ("", "redacted/job", False), ("rtsp", "other/job", False)])
async def test_hls_cannot_bypass_cdn_auth_through_anonymous_read(protocol, path, allowed):
    publisher = Publisher(None, None)
    assert await publisher.authorize(dict(action="read", protocol=protocol, path=path)) is allowed


@pytest.mark.parametrize("scheme", ["rtsp", "rtmp"])
async def test_complete_publish_address_keeps_path_query_and_credentials(scheme):
    settings = Settings(_env_file=None, source_allowlist=["127.0.0.1"])
    publisher = Publisher(None, settings)
    publisher.close = AsyncMock(return_value=True)
    publisher._change = AsyncMock(return_value=True)
    target = f"{scheme}://user:password@127.0.0.1/app/camera?sign=a%2Bb&key=1"
    assert await publisher.prepare(dict(id="job", publish_url=target), "owner", 2) == target


async def test_complete_publish_address_is_revalidated_by_worker():
    publisher = Publisher(None, Settings(_env_file=None))
    publisher.close = AsyncMock(return_value=True)
    publisher._change = AsyncMock(return_value=True)
    with pytest.raises(SourceError):
        await publisher.prepare(dict(id="job", publish_url="rtmp://127.0.0.1/live/test"), "owner", 1)


@pytest.mark.parametrize("address", ["http://127.0.0.1/live", "rtmp://127.0.0.1/live#fragment", "rtmp://169.254.169.254/live", "rtmp://10.0.0.2/live"])
async def test_invalid_publish_address_is_rejected_before_task_creation(address, tmp_path, monkeypatch):
    import httpx
    from vision_shield.api import routes
    from vision_shield.main import create_app

    monkeypatch.setattr(routes, "require_worker", AsyncMock())
    app = create_app(Settings(_env_file=None, media_root=tmp_path, source_allowlist=["127.0.0.1"]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        app.state.store.create = AsyncMock()
        response = await client.post("/v1/stream-tasks", json={"url": "rtsp://127.0.0.1/source", "publish_url": address})
        assert response.status_code == 422
        app.state.store.create.assert_not_awaited()
