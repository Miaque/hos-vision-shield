"""显式使用隔离 Redis，验证 Lua 的真实原子性。"""

import asyncio
import os

import pytest
from redis.asyncio import Redis
from unittest.mock import AsyncMock
import httpx
from redis.exceptions import ConnectionError as RedisConnectionError

from vision_shield.jobs.store import CapacityExceeded, IdempotencyConflict, TaskStore
from vision_shield.config import Settings
from vision_shield.main import create_app

pytestmark = pytest.mark.skipif(not os.getenv("TEST_REDIS_SERVER"), reason="需显式提供隔离 Redis 可执行文件")


@pytest.fixture
async def store(redis_url):
    """只清理本模块自己的隔离 Redis 测试库。"""
    client = Redis.from_url(redis_url, decode_responses=True)
    await client.flushdb()
    yield TaskStore(client, file_limit=2, stream_limit=2, notification_limit=2, lease_seconds=0.1)
    await client.aclose()


async def test_replay_is_before_capacity_and_conflict_does_not_overwrite(store):
    """并发重放只有一个任务，容量满仍可恢复旧结果。"""
    async def create():
        return await store.create("image", "upload", "private/input", idempotency_key="hub:1",
                                  fingerprint="digest", notification_target="event-hub")
    tasks = await asyncio.gather(*(create() for _ in range(12)))
    assert len({t["id"] for t in tasks}) == 1
    await store.create("video", "upload", "private/other", idempotency_key="hub:2",
                       fingerprint="other", notification_target="event-hub")
    assert (await create())["id"] == tasks[0]["id"]
    with pytest.raises(IdempotencyConflict):
        await store.create("image", "upload", "private/evil", idempotency_key="hub:1", fingerprint="changed")
    with pytest.raises(CapacityExceeded):
        await store.create("image", "upload", "private/new", idempotency_key="hub:3",
                           fingerprint="new", notification_target="event-hub")
    assert (await store.get(tasks[0]["id"]))["source_ref"] == "private/input"


async def test_target_namespaces_separate_environments_and_share_atomic_replays(store):
    """数据库克隆产生相同业务键时，各环境仍创建独立任务。"""
    async def create(target):
        return await store.create("image", "upload", "private/input", idempotency_key="same-key",
                                  fingerprint=target, notification_target=target)
    first, second = await asyncio.gather(create("dev.event-hub"), create("prod.event-hub"))
    assert first["id"] != second["id"]
    assert (await create("dev.event-hub"))["id"] == first["id"]
    assert (await store.replay("same-key", "dev.event-hub", notification_target="dev.event-hub"))["id"] == first["id"]
    assert await store.replay("same-key", "unused", notification_target="other.event-hub") is None
    assert first["submission_key"] == "same-key"
    assert store._binding_key('["dev.event-hub","same-key"]') != store._binding_key("same-key", "dev.event-hub")


@pytest.mark.parametrize("terminal", ["succeeded", "cancelled", "worker_lost"])
async def test_all_file_terminal_paths_persist_notification_with_state(store, terminal):
    """正常完成、排队取消和失租约均在同一 Lua 中登记通知。"""
    task = await store.create("video", "upload", "private/input", idempotency_key="hub:1",
                              fingerprint="digest", notification_target="event-hub")
    if terminal == "cancelled":
        await store.request_stop(task["id"], "cancelled")
    else:
        owned = await store.claim(task["id"], "worker")
        if terminal == "succeeded":
            await store.finish(task["id"], "worker", owned["generation"], "succeeded", "result.mp4")
        else:
            await asyncio.sleep(0.15)
            await store.recover_expired()
    current = await store.get(task["id"])
    claimed = await store.claim_notifications(limit=2)
    assert len(claimed) == 1
    assert claimed[0]["task"]["state"] == current["state"]
    assert claimed[0]["task"]["sequence"] == current["sequence"]
    assert await store.complete_notification(claimed[0], delivered=True)
    assert await store.claim_notifications(limit=2) == []
    assert await store.get(task["id"]) is not None


async def test_stream_old_ack_cannot_clear_new_state_or_release_lifecycle_slot(store):
    """直播首次 ready 确认不释放额度，过期确认不覆盖新状态。"""
    task = await store.create("stream", "rtsp", "rtsp://camera/live", idempotency_key="platform:1",
                              fingerprint="digest", notification_target="platform")
    owned = await store.claim(task["id"], "worker")
    await store.update_owned(task["id"], "worker", owned["generation"], state="running", ready=True)
    event = (await store.claim_notifications(limit=1))[0]
    await store.request_stop(task["id"], "stopped")
    assert not await store.complete_notification(event, delivered=True)
    assert await store.client.scard("vs:notifications:outstanding") == 1
    event = (await store.claim_notifications(limit=1))[0]
    await store.complete_notification(event, delivered=True)
    assert await store.client.scard("vs:notifications:outstanding") == 1
    await store.finish(task["id"], "worker", owned["generation"], "stopped")
    final = (await store.claim_notifications(limit=1))[0]
    await store.complete_notification(final, delivered=True)
    assert await store.client.scard("vs:notifications:outstanding") == 0


async def test_two_notifiers_share_target_limits_and_expired_lease_is_reclaimed(store):
    """一个离线目标不占用另一个目标的配额，旧令牌不能确认接管。"""
    store.notification_limit = 8
    for target in ("event-hub", "platform"):
        for index in range(3):
            task = await store.create("image", "upload", "private", idempotency_key=f"{target}:{index}",
                                      fingerprint="same", notification_target=target)
            await store.request_stop(task["id"], "cancelled")
    first = await store.claim_notifications()
    assert len(first) == 4
    assert await store.claim_notifications() == []
    event = first[0]
    task = event["task"]
    await store.client.hset(f"vs:notification:{task['id']}", "lease_until", 0)
    for key in ("vs:notifications:inflight", f"vs:notifications:inflight:{task['notification_target']}",
                f"vs:notifications:due:{task['notification_target']}"):
        await store.client.zadd(key, {task["id"]: 0})
    replacement = (await store.claim_notifications(limit=1, targets=(task["notification_target"],)))[0]
    assert replacement["token"] != event["token"]
    assert not await store.complete_notification(event, delivered=True)
    assert await store.complete_notification(replacement, delivered=True)


async def test_upload_race_and_lost_create_response_keep_only_owned_input(redis_url, tmp_path, monkeypatch):
    """同键并发输家不删除赢家，创建结果未知时保留已绑定输入。"""
    from vision_shield.api import routes
    monkeypatch.setattr(routes, "require_worker", AsyncMock())
    settings = Settings(_env_file=None, redis_url=redis_url, media_root=tmp_path,
        vision_shield_notification_amqp_url="amqp://mq/%2F", notification_targets={"event-hub": {"secret": "x" * 32}})
    app = create_app(settings)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        await app.state.redis.flushdb()
        headers = {"Idempotency-Key": "hub:upload", "X-Notification-Target": "event-hub"}
        async def upload(content=b"input"):
            return await client.post("/v1/file-tasks/uploads", headers=headers,
                                     data={"kind": "image"}, files={"file": ("a.png", content)})
        responses = await asyncio.gather(*(upload() for _ in range(8)))
        assert all(r.status_code == 202 for r in responses)
        identifier = responses[0].json()["id"]
        assert {r.json()["id"] for r in responses} == {identifier}
        assert [p.name for p in tmp_path.iterdir()] == [identifier]
        assert (await upload(b"changed")).status_code == 409
        assert len(list(tmp_path.iterdir())) == 1
        original_create = app.state.store.create
        async def lost_response(*args, **kwargs):
            await original_create(*args, **kwargs)
            raise RedisConnectionError("模拟创建已提交但响应丢失")
        app.state.store.create = lost_response
        headers["Idempotency-Key"] = "hub:lost"
        assert (await upload()).status_code == 503
        app.state.store.create = original_create
        response = await upload()
        assert response.status_code == 202
        assert (tmp_path / response.json()["id"] / "source.bin").read_bytes() == b"input"
        assert len(list(tmp_path.iterdir())) == 2
        # 同键终态恢复不依赖 worker 当前可用性，也不重新占接纳额度。
        await app.state.store.request_stop(identifier, "cancelled")
        headers["Idempotency-Key"] = "hub:upload"
        monkeypatch.setattr(routes, "require_worker", AsyncMock(side_effect=RuntimeError("不应调用")))
        assert (await upload()).json()["state"] == "cancelled"
        batch = await client.post("/v1/tasks/query", json={"task_ids": [identifier, identifier]})
        assert batch.status_code == 200 and len(batch.json()["items"]) == 1
        assert batch.json()["items"][0]["sequence"] > 1


async def test_output_is_fenced_and_stats_do_not_create_notifications(store):
    """当前代次实际发布才更新健康，冻结发布后的旧输出不能续期。"""
    task = await store.create("stream", "rtsp", "rtsp://camera/live", idempotency_key="platform:health",
                              fingerprint="same", notification_target="platform")
    owned = await store.claim(task["id"], "worker")
    generation = owned["generation"]
    await store.publisher_change(task["id"], "open", owner_id="worker", generation=generation)
    assert await store.mark_output(task["id"], "worker", generation)
    running = await store.update_owned(task["id"], "worker", generation, state="running", ready=True)
    await store.update_owned(task["id"], "worker", generation, stats={"processed_frames": 5})
    assert (await store.get(task["id"]))["sequence"] == running["sequence"]
    assert (await store.health(await store.get(task["id"])))["lease_valid"]
    await store.publisher_change(task["id"], "freeze", owner_id="worker", generation=generation)
    assert not await store.mark_output(task["id"], "worker", generation)
