"""隔离 Redis 与真实子进程验证外部推流的停流确认。"""

import asyncio
import json
import socket
import sys

import httpx
import pytest
from redis.asyncio import Redis

from tests.conftest import unused_port
from vision_shield.config import Settings
from vision_shield.jobs.reconcile import reconcile_once
from vision_shield.jobs.store import TaskStore
from vision_shield.main import create_app
from vision_shield.media.publisher import Publisher
from vision_shield.media.runner import run_process


@pytest.fixture(params=["rtsp", "rtmp"])
async def external_stream(redis_url, request):
    async with Redis.from_url(redis_url, decode_responses=True) as client:
        await client.flushdb()
        store = TaskStore(client)
        publisher = Publisher(store, Settings(_env_file=None, source_allowlist=["127.0.0.1"]))
        task = await store.create("stream", "rtsp", "rtsp://127.0.0.1/source",
                                  publish_url=f"{request.param}://127.0.0.1/app/camera")
        owned = await store.claim(task["id"], "owner")
        await publisher.prepare(owned, "owner", owned["generation"])
        await store.mark_output(task["id"], "owner", owned["generation"])
        await store.update_owned(task["id"], "owner", owned["generation"], state="running", ready=True)
        yield store, publisher, owned


async def expire(store, task):
    """只改变测试租约，恢复及确认仍执行真实 Lua。"""
    current = await store.get(task["id"])
    current["lease_until"] = 1
    await store.client.hset(f"vs:task:{task['id']}", "payload", json.dumps(current))


async def test_empty_media_sessions_cannot_confirm_external_stop(external_stream):
    store, publisher, owned = external_stream
    await store.request_stop(owned["id"], "stopped")
    await expire(store, owned)

    async def dispatch(task):
        pytest.fail("停止意图不得重新派发")

    await reconcile_once(store, dispatch, publisher.close)
    current = await store.get(owned["id"])
    assert current["state"] == "recovering" and not current["ready"]
    assert current["desired_state"] == "stopped" and not current["publish_allowed"]
    assert not (await store.health(current))["lease_valid"]
    assert not await store.publisher_change(owned["id"], "clear", sessions=[])
    assert (await store.get(owned["id"]))["state"] != "stopped"
    assert len(await store.active()) == 1


async def test_replacement_waits_for_confirmed_exit_and_rejects_stale_confirmation(external_stream):
    store, publisher, owned = external_stream
    await expire(store, owned)
    dispatched = []

    async def dispatch(task):
        dispatched.append(task["id"])

    await reconcile_once(store, dispatch, publisher.close)
    assert dispatched == []
    assert not await store.claim(owned["id"], "replacement")
    assert await publisher.close(owned, "owner", owned["generation"], process_exited=True)
    await reconcile_once(store, dispatch, publisher.close)
    assert dispatched == [owned["id"]]
    replacement = await store.claim(owned["id"], "replacement")
    assert replacement["generation"] == owned["generation"] + 1
    await publisher.prepare(replacement, "replacement", replacement["generation"])
    assert not await publisher.close(owned, "owner", owned["generation"], process_exited=True)
    current = await store.get(owned["id"])
    assert current["publish_allowed"] and current["publisher_owner_id"] == "replacement"


async def test_only_matching_process_exit_can_finish_stop_after_lease_expiry(external_stream):
    store, publisher, owned = external_stream
    await store.request_stop(owned["id"], "stopped")
    assert not await store.finish(owned["id"], "owner", owned["generation"], "stopped")
    await expire(store, owned)
    await store.recover_expired()
    assert not await publisher.close(owned, "other", owned["generation"], process_exited=True)
    assert not await publisher.close(owned, "owner", owned["generation"] + 1, process_exited=True)
    assert await publisher.close(owned, "owner", owned["generation"], process_exited=True)
    current = await store.get(owned["id"])
    assert current["state"] == "stopped" and not current["ready"]
    assert not current["publish_allowed"] and current["publisher_owner_id"] is None
    assert await store.active() == []


async def test_legacy_live_task_is_not_assumed_to_have_exited(external_stream):
    store, publisher, owned = external_stream
    current = await store.get(owned["id"])
    del current["publisher_owner_id"]
    current["lease_until"] = 1
    await store.client.hset(f"vs:task:{owned['id']}", "payload", json.dumps(current))
    await store.request_stop(owned["id"], "stopped")
    await store.recover_expired()
    assert not await publisher.close(owned)
    assert (await store.get(owned["id"]))["publisher_owner_id"] == "owner"
    assert await publisher.close(owned, "owner", owned["generation"], process_exited=True)
    assert (await store.get(owned["id"]))["state"] == "stopped"


async def test_legacy_queued_task_and_unpublished_stop_remain_safe(redis_url):
    async with Redis.from_url(redis_url, decode_responses=True) as client:
        await client.flushdb()
        store = TaskStore(client)
        publisher = Publisher(store, Settings(_env_file=None))
        task = await store.create("stream", "rtsp", "rtsp://127.0.0.1/source", publish_url="rtmp://127.0.0.1/app/camera")
        del task["publisher_owner_id"]
        await client.hset(f"vs:task:{task['id']}", "payload", json.dumps(task))
        claimed = []

        async def dispatch(record):
            claimed.append(await store.claim(record["id"], "owner"))

        await reconcile_once(store, dispatch, publisher.close)
        assert len(claimed) == 1 and claimed[0]["generation"] == 1
        await store.request_stop(task["id"], "stopped")
        assert await publisher.close(claimed[0])
        assert (await store.get(task["id"]))["state"] == "stopped"


async def test_unconfirmed_publisher_is_current_until_exit_allows_queueing(external_stream, redis_url):
    store, publisher, owned = external_stream
    await expire(store, owned)
    await store.recover_expired()
    app = create_app(Settings(_env_file=None, redis_url=redis_url))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        groups = (await client.get("/v1/tasks")).json()
        assert [task["id"] for task in groups["current"]] == [owned["id"]]
        assert groups["queued"] == []
        assert await publisher.close(owned, "owner", owned["generation"], process_exited=True)
        groups = (await client.get("/v1/tasks")).json()
        assert [task["id"] for task in groups["queued"]] == [owned["id"]]
        assert groups["current"] == []


async def test_default_rtsp_stop_still_uses_media_session_cleanup(redis_url):
    async with Redis.from_url(redis_url, decode_responses=True) as client:
        await client.flushdb()
        store = TaskStore(client)
        publisher = Publisher(store, Settings(_env_file=None))
        task = await store.create("stream", "rtsp", "rtsp://127.0.0.1/source")
        owned = await store.claim(task["id"], "owner")
        await publisher.prepare(owned, "owner", owned["generation"])
        await store.request_stop(task["id"], "stopped")
        await expire(store, task)
        await store.recover_expired()
        assert await publisher.close(task)
        assert (await store.get(task["id"]))["state"] == "stopped"


@pytest.mark.parametrize("cooperative", [True, False])
async def test_real_child_exit_is_confirmed_after_socket_closes(external_stream, cooperative):
    store, publisher, owned = external_stream
    port = unused_port()
    script = (
        "import socket,sys,time; "
        f"s=socket.socket(); s.bind(('127.0.0.1',{port})); s.listen(); "
        "print('{\"ready\":true}',flush=True); "
        + ("sys.stdin.readline(); sys.stdin.readline()" if cooperative else "time.sleep(30)")
    )
    cancel = asyncio.Event()
    confirmed = []

    async def event_received(event):
        if event.get("ready"):
            await store.request_stop(owned["id"], "stopped")
            await expire(store, owned)
            await store.recover_expired()
            assert not await publisher.close(owned)
            assert (await store.get(owned["id"]))["state"] != "stopped"
            cancel.set()

    async def child_exited():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        assert await publisher.close(owned, "owner", owned["generation"], process_exited=True)
        confirmed.append(True)

    result = await asyncio.wait_for(run_process([sys.executable, "-u", "-c", script], {}, cancel,
        event_received, stop_timeout=0.1, on_exit=child_exited), 10)
    assert result["cancelled"] and result["returncode"] is not None
    assert confirmed == [True] and (await store.get(owned["id"]))["state"] == "stopped"


async def test_failed_spawn_can_confirm_no_child_exists(tmp_path):
    confirmed = []

    async def on_exit():
        confirmed.append(True)

    async def on_event(event):
        pytest.fail("启动失败不应产生媒体事件")

    with pytest.raises(OSError):
        await run_process([str(tmp_path / "missing-program")], {}, asyncio.Event(), on_event, on_exit=on_exit)
    assert confirmed == [True]


@pytest.mark.parametrize("media_status,expected", [(200, 200), (503, 503)])
async def test_readiness_still_requires_mediamtx_for_external_streams(redis_url, monkeypatch, media_status, expected):
    app = create_app(Settings(_env_file=None, redis_url=redis_url))
    original_client = httpx.AsyncClient
    async with app.router.lifespan_context(app), original_client(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        await app.state.redis.flushdb()
        seconds, micros = await app.state.redis.time()
        for pool in ("files", "streams"):
            await app.state.redis.zadd(f"vs:workers:{pool}", {"worker": seconds * 1000 + micros // 1000})
        await app.state.store.create("stream", "rtsp", "rtsp://127.0.0.1/source", publish_url="rtmp://127.0.0.1/app/camera")
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(media_status)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
        assert (await client.get("/health/ready")).status_code == expected
        assert len(requests) == 1 and requests[0].url.path == "/v3/paths/list"
