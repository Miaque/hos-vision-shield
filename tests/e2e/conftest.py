import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from redis.asyncio import Redis

from tests.conftest import NO_WINDOW, ROOT, stop_process, unused_port


@pytest.fixture
async def native_app(redis_url, tmp_path):
    port = unused_port()
    rtsp_port, media_api_port = unused_port(), unused_port()
    env = dict(os.environ, REDIS_URL=redis_url, MEDIA_ROOT=str(tmp_path / "data"),
               SOURCE_ALLOWLIST='["127.0.0.1"]', RECONCILE_SECONDS="0.2", LEASE_SECONDS="2", HEARTBEAT_SECONDS="0.5",
               RTSP_PUBLIC_BASE_URL=f"rtsp://127.0.0.1:{rtsp_port}", MEDIAMTX_CONTROL_URL=f"http://127.0.0.1:{media_api_port}",
               FACE_MODEL_PATH=str(ROOT / "models/face.jit"), PLATE_MODEL_PATH=str(ROOT / "models/license_plate.jit"))
    media_binary = os.getenv("TEST_MEDIAMTX_BIN") or str(ROOT / ".tools/mediamtx/mediamtx.exe")
    media_config = yaml.safe_load((ROOT / "deploy/mediamtx.yml").read_text("utf-8"))
    media_config.update(rtspAddress=f"127.0.0.1:{rtsp_port}", apiAddress=f"127.0.0.1:{media_api_port}",
                        authHTTPAddress=f"http://127.0.0.1:{port}/internal/media/publish-check")
    media_config["authHTTPExclude"].append({"action": "publish", "path": "test-source"})
    config_path = tmp_path / "mediamtx.yml"
    config_path.write_text(yaml.safe_dump(media_config), "utf-8")
    commands = [
        [media_binary, str(config_path)],
        [sys.executable, "-m", "uvicorn", "vision_shield.main:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
        [sys.executable, "-m", "vision_shield.jobs.worker", "files"],
        [sys.executable, "-m", "vision_shield.jobs.reconcile"],
        [sys.executable, "-m", "vision_shield.jobs.worker", "streams"],
    ]
    processes, logs = [], []
    try:
        async with Redis.from_url(redis_url) as redis:
            await redis.flushdb()
        for index, command in enumerate(commands):
            log = (tmp_path / f"process-{index}.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=tmp_path if index == 0 else ROOT, env=env, stdout=log, stderr=log, creationflags=NO_WINDOW))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False) as client:
            async with asyncio.timeout(20):
                while True:
                    try:
                        async with Redis.from_url(redis_url) as redis:
                            ready = await redis.zcard("vs:workers:files") and await redis.zcard("vs:workers:streams")
                        if ready and (await client.get("/health/live")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    assert all(process.poll() is None for process in processes), [p.returncode for p in processes]
                    await asyncio.sleep(0.1)
            def spawn_stream_worker():
                log = (tmp_path / f"replacement-{len(processes)}.log").open("wb")
                logs.append(log)
                process = subprocess.Popen([sys.executable, "-m", "vision_shield.jobs.worker", "streams"],
                                           cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=NO_WINDOW)
                processes.append(process)
                return process

            yield dict(client=client, root=tmp_path / "data", processes=processes, rtsp_port=rtsp_port,
                       media_api_port=media_api_port, file_worker=processes[2],
                       stream_worker=processes[-1], spawn_stream_worker=spawn_stream_worker)
    finally:
        for process in reversed(processes):
            stop_process(process)
        for log in logs:
            log.close()
