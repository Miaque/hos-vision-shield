import asyncio
import io
import shutil

import httpx
from PIL import Image

from tests.conftest import ROOT


async def wait_terminal(client, task_id):
    async with asyncio.timeout(120):
        while True:
            task = (await client.get(f"/v1/tasks/{task_id}")).json()
            if task["state"] in {"succeeded", "failed", "cancelled", "stopped"}:
                return task
            await asyncio.sleep(0.2)


async def test_real_native_queue_and_image_model(native_app, http_media):
    client = native_app["client"]
    base, directory = http_media
    with (ROOT / "samples/official-demo.jpg").open("rb") as source:
        response = await client.post("/v1/file-tasks/uploads", data={"kind": "image"}, files={"file": ("face.jpg", source, "image/jpeg")})
    assert response.status_code == 202, response.text
    done = await wait_terminal(client, response.json()["id"])
    assert done["state"] == "succeeded", done
    output = await client.get(done["result_url"])
    assert output.status_code == 200
    with Image.open(io.BytesIO(output.content)) as result:
        assert result.size == (1080, 1920)
        assert all(abs(v - 64) < 10 for v in result.getpixel((560, 270)))
    assert list(native_app["root"].glob("*/source.bin"))
    created = await client.post("/v1/file-tasks", json={"kind": "image", "url": f"{base}/no-target.png"})
    done = await wait_terminal(client, created.json()["id"])
    assert done["state"] == "succeeded", done
    with Image.open(io.BytesIO((await client.get(done["result_url"])).content)) as no_target:
        assert no_target.getpixel((10, 10)) == (110, 160, 210)
    shutil.copyfile(ROOT / "samples/official-demo.jpg", directory / "face.jpg")
    remote = await client.post("/v1/file-tasks", json={"kind": "image", "url": f"{base}/face.jpg"})
    assert (await wait_terminal(client, remote.json()["id"]))["state"] == "succeeded"
    with (directory / "no-target.png").open("rb") as source:
        uploaded = await client.post("/v1/file-tasks/uploads", data={"kind": "image"}, files={"file": ("plain.png", source)})
    assert (await wait_terminal(client, uploaded.json()["id"]))["state"] == "succeeded"
    damaged = await client.post("/v1/file-tasks/uploads", data={"kind": "image"}, files={"file": ("bad.jpg", b"damaged image")})
    failed = await wait_terminal(client, damaged.json()["id"])
    assert failed["state"] == "failed"
    assert (await client.get(f"/v1/tasks/{failed['id']}/result")).status_code == 409
