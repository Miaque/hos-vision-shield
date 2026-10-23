"""部署停止只回收在途媒体，不伪造用户终止意图或成功。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vision_shield.config import Settings
from vision_shield.jobs import tasks


@pytest.mark.parametrize("fails", [False, True])
async def test_deployment_shutdown_leaves_unfinished_file_for_lease_recovery(tmp_path, monkeypatch, fails):
    stop = asyncio.Event()
    broker = SimpleNamespace(connection_pool=None, state=SimpleNamespace(stop_event=stop), ack_message=AsyncMock())
    record = dict(id="job", kind="image", generation=1, source_kind="upload", source_ref="local", desired_state="run")
    store = SimpleNamespace(claim=AsyncMock(return_value=record), get=AsyncMock(return_value=record),
                            heartbeat=AsyncMock(return_value=True), finish=AsyncMock(), update_owned=AsyncMock())
    monkeypatch.setattr(tasks, "Settings", lambda: Settings(_env_file=None, media_root=tmp_path, heartbeat_seconds=0.01))
    monkeypatch.setattr(tasks, "TaskStore", lambda *a: store)
    monkeypatch.setattr(tasks, "Redis", lambda **kw: None)

    async def run(command, payload, cancel, on_event, timeout):
        stop.set()
        await asyncio.wait_for(cancel.wait(), 0.5)
        if fails:
            raise RuntimeError("受控停止时的媒体异常")
        return dict(cancelled=True, returncode=3)

    monkeypatch.setattr(tasks, "run_process", run)
    await tasks.execute("job", broker, SimpleNamespace(message=SimpleNamespace(task_id="message")))
    store.finish.assert_not_called()
    assert record["desired_state"] == "run"


async def test_stopping_worker_does_not_claim_another_task(monkeypatch):
    stop = asyncio.Event()
    stop.set()
    broker = SimpleNamespace(state=SimpleNamespace(stop_event=stop))
    monkeypatch.setattr(tasks, "Redis", lambda **kw: pytest.fail("停止后不得领取新任务"))
    await tasks.execute("job", broker, None)


@pytest.mark.parametrize('profile_mode', [True, 'host', 'forward-primary-face', 'forward-detail-plate'])
@pytest.mark.parametrize('interval,tracking', [(1, 'template'), (3, 'template'), (3, 'flow')])
async def test_requested_device_profile_passes_to_child_and_is_stored_without_stats(tmp_path, monkeypatch, interval, tracking, profile_mode):
    broker = SimpleNamespace(connection_pool=None, state=SimpleNamespace(), ack_message=AsyncMock())
    record = dict(id='job', kind='video', generation=1, source_kind='upload', source_ref='local',
                  desired_state='run', profile_detection=profile_mode, detection_interval=interval, tracking=tracking)
    store = SimpleNamespace(claim=AsyncMock(return_value=record), get=AsyncMock(return_value=record),
                            heartbeat=AsyncMock(return_value=True), finish=AsyncMock(), update_owned=AsyncMock())
    monkeypatch.setattr(tasks, 'Settings', lambda: Settings(_env_file=None, media_root=tmp_path))
    monkeypatch.setattr(tasks, 'TaskStore', lambda *a: store)
    monkeypatch.setattr(tasks, 'Redis', lambda **kw: None)
    profile = dict(state='complete', samples=[dict(frame=64)])
    counts = dict(full_frames=4, skipped_frames=8, local_rechecks=2)
    async def run(command, payload, cancel, on_event, timeout):
        assert payload['profile_detection'] == profile_mode and 'profile_detection' not in payload['config']
        assert payload['config']['detection_interval'] == interval
        assert payload['config']['tracking'] == tracking
        await on_event(dict(device_profile=profile))
        await on_event(dict(detection_stats=counts))
        return dict(returncode=0, result_ref=str(tmp_path / 'job/attempt-1/result.mp4'))
    monkeypatch.setattr(tasks, 'run_process', run)
    await tasks.execute('job', broker, SimpleNamespace(message=SimpleNamespace(task_id='message')))
    assert [call.kwargs for call in store.update_owned.call_args_list] == [dict(device_profile=profile), dict(detection_stats=counts)]
    assert store.finish.call_args.args[3] == 'succeeded'


@pytest.mark.parametrize("mode", [None, "solid", "mosaic"])
async def test_remote_file_download_keeps_deployment_event_and_finishes(tmp_path, monkeypatch, mode):
    stop = asyncio.Event()
    broker = SimpleNamespace(connection_pool=None, state=SimpleNamespace(stop_event=stop), ack_message=AsyncMock())
    record = dict(id="job", kind="image", generation=1, source_kind="url", source_ref="http://approved/input", desired_state="run")
    store = SimpleNamespace(claim=AsyncMock(return_value=record), get=AsyncMock(return_value=record),
                            heartbeat=AsyncMock(return_value=True), finish=AsyncMock(), update_owned=AsyncMock())
    monkeypatch.setattr(tasks, "Settings", lambda: Settings(_env_file=None, media_root=tmp_path, heartbeat_seconds=0.01))
    monkeypatch.setattr(tasks, "TaskStore", lambda *a: store)
    monkeypatch.setattr(tasks, "Redis", lambda **kw: None)
    monkeypatch.setattr(tasks, "download", AsyncMock())
    if mode is not None:
        record["redaction_mode"] = mode
    async def run(command, payload, cancel, on_event, timeout):
        assert payload["config"]["redaction_mode"] == (mode or "solid")
        await on_event(dict(stats=dict(processed_frames=1, dropped_frames=0, elapsed_seconds=2),
                            timings=dict(frame=1.5, inference=1.25, encode=0.1), video_encoder="h264_nvenc", video_decoder="h264_nvdec"))
        await asyncio.sleep(0.04)
        assert not cancel.is_set()
        return dict(returncode=0, result_ref=str(tmp_path / "job/attempt-1/result.png"))
    monkeypatch.setattr(tasks, "run_process", run)
    await tasks.execute("job", broker, SimpleNamespace(message=SimpleNamespace(task_id="message")))
    assert store.finish.call_args.args[3] == "succeeded"
    assert store.heartbeat.await_count >= 2
    assert store.update_owned.call_args.kwargs == dict(
        stats=dict(processed_frames=1, dropped_frames=0, elapsed_seconds=2),
        timings=dict(frame=1.5, inference=1.25, encode=0.1), video_encoder="h264_nvenc", video_decoder="h264_nvdec")


async def test_reconciler_cancellation_closes_brokers_and_client(monkeypatch):
    from vision_shield.jobs import broker, reconcile
    files = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())
    streams = SimpleNamespace(startup=AsyncMock(), shutdown=AsyncMock())
    client = SimpleNamespace(eval=AsyncMock(side_effect=asyncio.CancelledError), aclose=AsyncMock())
    monkeypatch.setattr(broker, "file_broker", files)
    monkeypatch.setattr(broker, "stream_broker", streams)
    monkeypatch.setattr(reconcile, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(reconcile, "Redis", SimpleNamespace(from_url=lambda *a, **kw: client))
    with pytest.raises(asyncio.CancelledError):
        await reconcile.main()
    files.shutdown.assert_awaited_once()
    streams.shutdown.assert_awaited_once()
    client.aclose.assert_awaited_once()
