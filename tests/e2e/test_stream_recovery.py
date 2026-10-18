import asyncio
import subprocess
import sys
import time
import json
import pytest

import av
from redis.asyncio import Redis

from tests.conftest import NO_WINDOW, ROOT, stop_process


async def wait_state(client, task_id, states):
    async with asyncio.timeout(90):
        while True:
            task = (await client.get(f"/v1/tasks/{task_id}")).json()
            assert task["state"] != "failed", task
            if task["state"] in states:
                return task
            await asyncio.sleep(0.2)


def read_frame(url):
    with av.open(url, options={"rtsp_transport": "tcp", "analyzeduration": "100000", "probesize": "32768"}, timeout=(10, 10)) as reader:
        return next(reader.decode(video=0)).to_ndarray(format="rgb24")


@pytest.mark.parametrize("attempt", range(3))
async def test_native_rtsp_playback_reconnect_worker_restart_and_stop(native_app, redis_url, attempt):
    client = native_app["client"]
    source_url = f"rtsp://127.0.0.1:{native_app['rtsp_port']}/test-source"
    sources = []
    width, height = (1728, 536) if attempt == 1 else (320, 240)

    def start_source():
        command = ["ffmpeg", "-v", "error", "-re", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=4"]
        command += ["-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-c:a", "aac"] if attempt == 1 else ["-an"]
        command += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "4",
                    "-f", "rtsp", "-rtsp_transport", "tcp", source_url]
        process = subprocess.Popen(command,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
        )
        sources.append(process)
        return process

    def replace_worker():
        native_app["stream_worker"] = native_app["spawn_stream_worker"]()

    try:
        source = start_source()
        await asyncio.sleep(0.8)
        created = await client.post("/v1/stream-tasks", json={"url": source_url})
        assert created.status_code == 202, created.text
        task_id = created.json()["id"]
        task = await wait_state(client, task_id, {"running"})
        pixels = await asyncio.wait_for(asyncio.to_thread(read_frame, task["playback_url"]), 30)
        assert pixels.shape == (height, width, 3)
        stop_process(source)
        await wait_state(client, task_id, {"recovering"})
        source = start_source()
        await wait_state(client, task_id, {"running"})
        stop_process(native_app["stream_worker"])
        replace_worker()
        async with Redis.from_url(redis_url) as redis, asyncio.timeout(90):
            while json.loads(await redis.hget(f"vs:task:{task_id}", "payload"))["generation"] < 2:
                await asyncio.sleep(0.1)
        await wait_state(client, task_id, {"running"})
        # Race durable stop against a fresh receiver and lease recovery.
        stop_process(native_app["stream_worker"])
        replace_worker()
        requested = await client.patch(f"/v1/tasks/{task_id}", json={"desired_state": "stopped"})
        assert requested.status_code in {200, 202}
        stopped = await wait_state(client, task_id, {"stopped"})
        assert not stopped["ready"]
        stop_process(native_app["stream_worker"])
        replace_worker()
        await asyncio.sleep(3)
        assert (await client.get(f"/v1/tasks/{task_id}")).json()["state"] == "stopped"
        output = await client.get(f"http://127.0.0.1:{native_app['media_api_port']}/v3/paths/get/redacted/{task_id}")
        assert output.status_code == 404 or not output.json().get("ready")
    finally:
        for source in sources:
            stop_process(source)
