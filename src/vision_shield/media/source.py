"""Bounded remote input with DNS pinned to the addresses actually authorized."""

import asyncio
import ipaddress
import socket
from pathlib import Path

import httpx
from vision_shield.config import Settings


class SourceError(Exception):
    def __init__(self, code="source_not_allowed", status=422):
        self.code, self.status = code, status
        super().__init__(code)


async def validate_source(value: str, allowlist: list[str], schemes: tuple[str, ...] = ("http", "https")) -> tuple[httpx.URL, str]:
    """Return the original URL and one approved IP; every resolved IP must be allowed."""
    try:
        url = httpx.URL(value)
        if url.scheme not in schemes or not url.host or url.fragment:
            raise SourceError()
        if url.scheme != "rtsp" and url.userinfo:
            raise SourceError()
        addresses = await asyncio.get_running_loop().getaddrinfo(url.host, url.port or (443 if url.scheme == "https" else 554 if url.scheme == "rtsp" else 80), type=socket.SOCK_STREAM)
        networks = []
        for item in allowlist:
            try:
                networks.append(ipaddress.ip_network(item))
            except ValueError:
                pass
        approved_host = url.host.lower() in {item.lower() for item in allowlist}
        permitted = []
        for address in {item[4][0] for item in addresses}:
            ip = ipaddress.ip_address(address)
            explicit_ip = any(ip in network for network in networks)
            if ip.is_multicast or ip.is_unspecified or ip.is_link_local:
                raise SourceError()
            if not explicit_ip and (not approved_host or not ip.is_global):
                raise SourceError()
            permitted.append(address)
        if not permitted:
            raise SourceError()
        return url, sorted(permitted)[0]
    except socket.gaierror as exc:
        raise SourceError("source_unreachable", 422) from exc
    except (ValueError, httpx.InvalidURL) as exc:
        raise SourceError() from exc


async def download(value: str, destination: Path, settings: Settings, kind: str) -> Path:
    """Stream approved HTTP content into this task's fixed destination with byte/time bounds."""
    limit = settings.max_image_bytes if kind == "image" else settings.max_video_bytes
    timeout = httpx.Timeout(settings.http_read_timeout, connect=settings.http_connect_timeout)
    try:
        async with asyncio.timeout(settings.download_timeout), httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            for redirect in range(4):
                original, address = await validate_source(value, settings.source_allowlist)
                pinned = original.copy_with(host=address)
                async with client.stream("GET", pinned, headers={"Host": original.netloc.decode()},
                                         extensions={"sni_hostname": original.host}) as response:
                    if response.is_redirect:
                        if redirect == 3 or "location" not in response.headers:
                            raise SourceError()
                        value = str(original.join(response.headers["location"]))
                        continue
                    if response.status_code != 200:
                        raise SourceError("source_unreachable", 422)
                    output = await asyncio.to_thread(destination.open, "wb")
                    size = 0
                    try:
                        async for block in response.aiter_bytes(65536):
                            size += len(block)
                            if size > limit:
                                raise SourceError("media_too_large", 413)
                            await asyncio.to_thread(output.write, block)
                    finally:
                        await asyncio.to_thread(output.close)
                    return destination
    except (httpx.HTTPError, TimeoutError) as exc:
        raise SourceError("source_unreachable", 422) from exc
