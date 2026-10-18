import pytest
from redis.asyncio import Redis

from vision_shield.jobs.reconcile import reconcile_once
from vision_shield.jobs.store import TaskStore


async def test_unsent_intent_survives_dispatch_failure(redis_url):
    async with Redis.from_url(redis_url) as client:
        await client.flushdb()
        store = TaskStore(client)
        task = await store.create("image", "url", "http://approved/image.png")

        async def unavailable(record):
            raise ConnectionError("broker unavailable")

        with pytest.raises(ConnectionError):
            await reconcile_once(store, unavailable)
        assert (await store.get(task["id"]))["dispatch_pending"]
        sent = []

        async def send(record):
            sent.append(record["id"])

        await reconcile_once(store, send)
        assert sent == [task["id"]]
        await store.request_stop(task["id"], "cancelled")
        sent.clear()
        await reconcile_once(store, send)
        assert sent == []
