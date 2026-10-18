import subprocess
import asyncio
import shutil

import av

from tests.conftest import NO_WINDOW, stop_process
from tests.e2e.test_images import wait_terminal


async def test_native_video_queue_and_model(native_app, generated_media, tmp_path, http_media):
    client = native_app["client"]
    source = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(generated_media / "silent.mp4"),
                    "-frames:v", "3", "-c:v", "libx264", str(source)],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    with source.open("rb") as media:
        created = await client.post("/v1/file-tasks/uploads", data={"kind": "video"}, files={"file": ("tiny.mp4", media, "video/mp4")})
    assert created.status_code == 202
    done = await wait_terminal(client, created.json()["id"])
    assert done["state"] == "succeeded", done
    result = await client.get(done["result_url"])
    target = tmp_path / "redacted.mp4"
    target.write_bytes(result.content)
    with av.open(str(target)) as video:
        assert len(list(video.decode(video=0))) == 3
    base, directory = http_media
    shutil.copyfile(source, directory / "tiny.mp4")
    created = await client.post("/v1/file-tasks", json={"kind": "video", "url": f"{base}/tiny.mp4"})
    done = await wait_terminal(client, created.json()["id"])
    assert done["state"] == "succeeded", done
    assert (await client.get(done["result_url"])).status_code == 200


async def test_active_cancel_and_worker_loss(native_app, generated_media):
    client = native_app["client"]

    async def submit_processing():
        with (generated_media / "silent.mp4").open("rb") as media:
            created = await client.post("/v1/file-tasks/uploads", data={"kind": "video"}, files={"file": ("video.mp4", media)})
        task_id = created.json()["id"]
        async with asyncio.timeout(30):
            while (await client.get(f"/v1/tasks/{task_id}")).json()["state"] != "processing":
                await asyncio.sleep(0.05)
        return task_id

    task_id = await submit_processing()
    response = await client.patch(f"/v1/tasks/{task_id}", json={"desired_state": "cancelled"})
    assert response.status_code == 202
    assert (await wait_terminal(client, task_id))["state"] == "cancelled"
    assert (await client.get(f"/v1/tasks/{task_id}/result")).status_code == 409
    lost_id = await submit_processing()
    stop_process(native_app["file_worker"])
    failed = await wait_terminal(client, lost_id)
    assert failed["state"] == "failed" and failed["error"]["code"] == "worker_lost"
