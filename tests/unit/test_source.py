import pytest
import asyncio
import socket
import httpx

from vision_shield.config import Settings
from vision_shield.media.source import SourceError, validate_source, download


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/a", "http://user:secret@camera/a"])
async def test_unapproved_or_non_http_sources_are_rejected(url):
    with pytest.raises(SourceError):
        await validate_source(url, [])


async def test_explicit_loopback_can_be_used_for_authorized_local_media():
    original, address = await validate_source("http://127.0.0.1:8001/a", ["127.0.0.1"])
    assert address == "127.0.0.1"
    assert original.host == "127.0.0.1"


async def test_all_resolved_addresses_must_be_authorized(monkeypatch):
    async def resolve(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in ["8.8.8.8", "127.0.0.1"]]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    with pytest.raises(SourceError):
        await validate_source("https://approved.test/image", ["approved.test"])


async def test_download_pins_connection_preserves_host_and_revalidates_redirect(monkeypatch, tmp_path):
    requests = []

    async def resolve(host, *args, **kwargs):
        address = "8.8.8.8" if host == "approved.test" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

    async def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(SourceError):
        await download("https://approved.test/image", tmp_path / "media", Settings(source_allowlist=["approved.test"], _env_file=None), "image")
    assert len(requests) == 1
    assert requests[0].url.host == "8.8.8.8"
    assert requests[0].headers["Host"] == "approved.test"
    assert requests[0].extensions["sni_hostname"] == "approved.test"


async def test_download_size_limit_uses_actual_body(http_media, tmp_path):
    base, _ = http_media
    settings = Settings(source_allowlist=["127.0.0.1"], max_image_bytes=10, _env_file=None)
    with pytest.raises(SourceError) as caught:
        await download(f"{base}/no-target.png", tmp_path / "source", settings, "image")
    assert caught.value.code == "media_too_large"


async def test_total_download_timeout_closes_request(monkeypatch, tmp_path):
    async def respond(request):
        await asyncio.sleep(1)
        return httpx.Response(200, content=b"media")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    settings = Settings(source_allowlist=["127.0.0.1"], download_timeout=0.01, _env_file=None)
    with pytest.raises(SourceError) as caught:
        await download("http://127.0.0.1/a", tmp_path / "source", settings, "image")
    assert caught.value.code == "source_unreachable"
