import asyncio

import pytest
from redis.asyncio import Redis

from vision_shield.jobs.store import CapacityExceeded, TaskStore


@pytest.fixture
async def store(redis_url):
    client = Redis.from_url(redis_url, decode_responses=True)
    # redis_url belongs to the isolated server created by conftest, never user data.
    await client.flushdb()
    yield TaskStore(client, file_limit=2, stream_limit=1, lease_seconds=0.1)
    await client.aclose()


async def test_admission_is_atomic_and_full_queue_does_not_leave_orphan(store):
    async def create():
        try:
            return await store.create("image", "url", "http://approved/image.png")
        except CapacityExceeded:
            return None
    results = await asyncio.gather(*(create() for _ in range(8)))
    accepted = [task for task in results if task]
    assert len(accepted) == 2
    assert len(await store.active()) == 2
    assert all(task["dispatch_pending"] for task in accepted)


async def test_only_one_claim_and_old_owner_cannot_change_result(store):
    task = await store.create("image", "url", "http://approved/image.png")
    claims = await asyncio.gather(*(store.claim(task["id"], f"worker-{i}") for i in range(8)))
    winners = [result for result in claims if result]
    assert len(winners) == 1
    owner = winners[0]
    assert not await store.finish(task["id"], "other", owner["generation"], "succeeded", "bad.jpg")
    assert await store.finish(task["id"], owner["owner_id"], owner["generation"], "succeeded", "result.jpg")
    assert not await store.finish(task["id"], owner["owner_id"], owner["generation"], "succeeded", "overwrite.jpg")
    assert (await store.get(task["id"]))["result_ref"] == "result.jpg"
    assert await store.active() == []


async def test_cancellation_wins_over_late_success_and_releases_capacity_once(store):
    task = await store.create("video", "upload", "data/test/source.mp4")
    owner = await store.claim(task["id"], "worker")
    requested = await store.request_stop(task["id"], "cancelled")
    assert requested["desired_state"] == "cancelled" and requested["state"] == "processing"
    assert await store.heartbeat(task["id"], "worker", owner["generation"])
    assert not await store.finish(task["id"], "worker", owner["generation"], "succeeded", "bad.mp4")
    assert await store.finish(task["id"], "worker", owner["generation"], "cancelled")
    await store.request_stop(task["id"], "cancelled")
    assert (await store.get(task["id"]))["result_ref"] is None
    assert await store.active() == []


async def test_expired_file_fails_but_stream_is_recoverable_and_stop_is_durable(store):
    file = await store.create("image", "url", "http://approved/image.png")
    stream = await store.create("stream", "rtsp", "rtsp://approved/live")
    await store.claim(file["id"], "lost-file")
    old = await store.claim(stream["id"], "lost-stream")
    await asyncio.sleep(0.15)
    await store.recover_expired()
    assert (await store.get(file["id"]))["error"]["code"] == "worker_lost"
    current = await store.claim(stream["id"], "replacement")
    assert current and current["generation"] > old["generation"]
    assert not await store.heartbeat(stream["id"], "lost-stream", old["generation"])
    await store.request_stop(stream["id"], "stopped")
    assert not await store.claim(stream["id"], "yet-another")
    assert not await store.finish(stream["id"], "replacement", current["generation"], "running")


async def test_stopped_queued_stream_never_claims_and_history_is_retained(store):
    task = await store.create("stream", "rtsp", "rtsp://approved/live")
    await store.request_stop(task["id"], "stopped")
    assert (await store.get(task["id"]))["state"] == "stopped"
    assert not await store.claim(task["id"], "worker")
    assert await store.active() == []
    assert await store.get(task["id"]) is not None
