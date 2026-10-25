"""监督方必须确认探测成功事件、正常退出和超时回收。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vision_shield.config import Settings
from vision_shield.jobs import worker


@pytest.mark.parametrize("result", [{"returncode": 0}, {"probe": {}, "returncode": 1},
                                   {"probe": {}, "returncode": 0, "cancelled": True}])
async def test_incomplete_probe_cannot_start_worker(monkeypatch, result):
    async def run(*args):
        return result
    monkeypatch.setattr(worker, "run_process", run)
    with pytest.raises(RuntimeError, match="预检"):
        await worker.probe_models(Settings(_env_file=None), asyncio.Event())


async def test_probe_success_and_supervisor_timeout(monkeypatch):
    async def success(command, payload, cancel, callback, timeout):
        assert payload["kind"] == "probe" and "directory" not in payload
        return dict(probe={"device": "cpu"}, returncode=0)
    monkeypatch.setattr(worker, "run_process", success)
    await worker.probe_models(Settings(_env_file=None), asyncio.Event())
    reclaimed = asyncio.Event()
    async def stuck(*args):
        try:
            await asyncio.Event().wait()
        finally:
            reclaimed.set()
    monkeypatch.setattr(worker, "run_process", stuck)
    monkeypatch.setattr(worker, "PROBE_TIMEOUT", 0.01)
    with pytest.raises(RuntimeError, match="超时"):
        await worker.probe_models(Settings(_env_file=None), asyncio.Event())
    assert reclaimed.is_set()


async def test_worker_waits_for_probe_and_stops_saturated_receiver(monkeypatch):
    stop, active, reclaimed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sampling, sampling_stopped = asyncio.Event(), asyncio.Event()
    calls = []
    broker = SimpleNamespace(state=SimpleNamespace(), connection_pool=None, get_all_tasks=lambda: {},
                             startup=AsyncMock(), shutdown=AsyncMock(), middlewares=[])
    async def listen():
        yield b"message"
        await asyncio.Event().wait()
    broker.listen = listen
    client = SimpleNamespace(time=AsyncMock(return_value=(1, 0)), zadd=AsyncMock(), zrem=AsyncMock(), aclose=AsyncMock())
    async def probe(settings, stopping, session):
        calls.append("probe")
        assert not broker.startup.called and not client.zadd.called
    async def observe(resource_client, pool, worker_id, device):
        assert resource_client is client and pool == 'files' and device == 'cpu'
        sampling.set()
        try:
            await asyncio.Event().wait()
        finally:
            sampling_stopped.set()
    receiver_class = worker.Receiver
    def receiver(*args, **kwargs):
        assert calls == ["probe"]
        assert isinstance(broker.state.media_process, worker.MediaProcess)
        assert kwargs['max_async_tasks'] == 1 and kwargs['max_prefetch'] == 0
        result = receiver_class(*args, **kwargs)
        async def callback(message, raise_err=False):
            active.set()
            await broker.state.stop_event.wait()
            await asyncio.sleep(0.01)
            reclaimed.set()
        result.callback = callback
        return result
    monkeypatch.setattr(worker, "Settings", lambda: Settings(_env_file=None, heartbeat_seconds=0.01))
    monkeypatch.setattr(worker, "file_broker", broker)
    monkeypatch.setattr(worker, "Redis", lambda **kw: client)
    monkeypatch.setattr(worker, "probe_models", probe)
    monkeypatch.setattr(worker, "observe_resources", observe)
    monkeypatch.setattr(worker, "Receiver", receiver)
    running = asyncio.create_task(worker.main("files", stop))
    try:
        await asyncio.wait_for(active.wait(), 1)
        stop.set()
        await asyncio.wait_for(running, 2)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert reclaimed.is_set()
    assert sampling.is_set() and sampling_stopped.is_set()
    client.zrem.assert_awaited_once()
    assert client.zrem.call_args.args[0] == "vs:workers:files"
    broker.shutdown.assert_awaited_once()


async def test_failed_probe_never_constructs_receiver_or_announces(monkeypatch):
    async def fails(*args):
        raise RuntimeError("模型预检失败")
    monkeypatch.setattr(worker, "probe_models", fails)
    monkeypatch.setattr(worker, "Redis", lambda **kw: pytest.fail("预检失败不得连接心跳存储"))
    monkeypatch.setattr(worker, "Receiver", lambda *a, **kw: pytest.fail("预检失败不得开始接单"))
    with pytest.raises(RuntimeError, match="预检"):
        await worker.main("files")
