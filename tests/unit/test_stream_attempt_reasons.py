import asyncio
import copy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.config import Settings


@pytest.mark.parametrize('debug', [False, True])
async def test_stream_diagnostics_require_debug_without_disabling_recovery(monkeypatch, caplog, debug):
    from vision_shield.jobs import tasks

    caplog.set_level('INFO', logger=tasks.__name__)
    record = dict(id='job', kind='stream', desired_state='run', generation=1, source_ref='rtsp://input')
    cancel = asyncio.Event()
    store = SimpleNamespace(update_owned=AsyncMock(return_value=True), finish=AsyncMock(),
                            get=AsyncMock(return_value=record))
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtsp://output'), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))

    async def run(*args, **kwargs):
        await args[3](dict(failure_type='RuntimeError', failure_at=['debug-site:123'],
            timings=dict(frame=1), video_encoder='h264_nvenc', detection_stats=dict(full_frames=1),
            stats=dict(processed_frames=1, stream=dict(attempt_id='sampled', recent_outputs=[])),
            stream_end=dict(reason='source_eof', phase='source_read')))
        return dict(returncode=1, retryable=True)

    async def retry_wait(awaitable, timeout):
        awaitable.close()
        cancel.set()
        raise TimeoutError

    monkeypatch.setattr(tasks, 'run_process', run)
    monkeypatch.setattr(tasks.asyncio, 'wait_for', retry_wait)
    await tasks.run_stream(record, store, Settings(_env_file=None, debug=debug), cancel, 'owner', 1)
    updates = [call.kwargs for call in store.update_owned.call_args_list]
    assert any(row.get('state') == 'recovering' for row in updates)
    assert publisher.close.await_count >= 2
    assert any('stream_attempts' in row for row in updates) is debug
    assert any('media_failure' in row for row in updates) is debug
    assert any('timings' in row for row in updates) is debug
    assert any('stream' in row.get('stats', {}) for row in updates) is debug
    assert ('debug-site:123' in caplog.text) is debug
    assert ('流脱敏连接结束' in caplog.text) is debug


def test_production_failure_details_do_not_collect_stack_or_ffmpeg_logs(monkeypatch):
    import av
    from vision_shield.media import attempts

    monkeypatch.setattr(attempts.traceback, 'extract_tb', lambda *args: pytest.fail('No traceback sampling'))
    error = av.error.InvalidDataError(1094995529, 'diagnostic detail', log=(16, 'rtsp', 'diagnostic log'))
    result = attempts.failure_details(error, 'source_open')
    assert result == dict(reason='source_open_error', phase='source_open',
                          error_type='InvalidDataError', errno=1094995529)


@pytest.mark.parametrize('output_case', ['stable', 'brief', 'unconfirmed', 'ready_only'])
async def test_reconnect_backoff_resets_only_after_sustained_confirmed_output(monkeypatch, output_case):
    from vision_shield.jobs import tasks

    record = dict(id='job', kind='stream', desired_state='run', generation=1, source_ref='rtsp://input')
    store = SimpleNamespace(update_owned=AsyncMock(return_value=True), finish=AsyncMock(), get=AsyncMock(return_value=record),
                            mark_output=AsyncMock(return_value=output_case != 'unconfirmed'))
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtsp://output'), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    cancel, waits, calls = asyncio.Event(), [], 0
    settings = Settings(_env_file=None)

    async def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            if output_case == 'ready_only':
                await args[3](dict(ready=True))
            else:
                first = tasks.time.monotonic()
                gap = 1 if output_case == 'brief' else settings.heartbeat_seconds
                for at in (first, first + gap):
                    await args[3](dict(output=True, ready=True, output_monotonic=at))
            await args[3](dict(stream_end=dict(reason='source_eof', phase='source_read')))
            return dict(returncode=0, retryable=True)
        await args[3](dict(stream_end=dict(reason='source_open_error', phase='source_open')))
        return dict(returncode=1, retryable=True)

    async def retry_wait(awaitable, timeout):
        awaitable.close()
        waits.append(timeout)
        if len(waits) == 7:
            cancel.set()
        raise TimeoutError

    monkeypatch.setattr(tasks, 'run_process', run)
    monkeypatch.setattr(tasks.asyncio, 'wait_for', retry_wait)
    await tasks.run_stream(record, store, settings, cancel, 'owner', 1)
    expected = [1, 2, 4, 8, 16] + ([1, 2] if output_case == 'stable' else [30, 30])
    assert waits == expected
    if output_case == 'ready_only':
        store.mark_output.assert_not_awaited()
    store.finish.assert_not_awaited()


async def test_eof_and_internal_publish_failure_have_distinct_retained_reasons(monkeypatch, caplog):
    from vision_shield.jobs import tasks

    caplog.set_level('INFO', logger=tasks.__name__)

    record = dict(id='job', kind='stream', desired_state='run', generation=1, source_ref='rtsp://private:secret@input')
    store = SimpleNamespace(update_owned=AsyncMock(return_value=True), get=AsyncMock(return_value=record), finish=AsyncMock())
    publisher = SimpleNamespace(prepare=AsyncMock(side_effect=['rtsp://output',
        httpx.ReadTimeout('rtsp://private:secret@control')]), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    cancel = asyncio.Event()
    saved = []

    async def update(*args, **kwargs):
        saved.append(copy.deepcopy(kwargs))
        return True

    store.update_owned.side_effect = update
    calls = 0

    async def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            event = dict(stream_end=dict(reason='source_eof', phase='source_read', attempt_id='first'))
            await args[3](event)
            return dict(returncode=0, reconnect=True, **event)
        raise httpx.ReadTimeout('rtsp://private:secret@control')

    async def stop_retry(awaitable, timeout):
        awaitable.close()
        if publisher.prepare.await_count == 2:
            cancel.set()
        raise TimeoutError

    monkeypatch.setattr(tasks, 'run_process', run)
    monkeypatch.setattr(tasks.asyncio, 'wait_for', stop_retry)
    await tasks.run_stream(record, store, Settings(_env_file=None, debug=True), cancel, 'owner', 1)
    history = next(value['stream_attempts'] for value in reversed(saved) if 'stream_attempts' in value)
    assert len(history) == 2
    assert history[0]['reason'] == 'source_eof' and history[0]['attempt_id'] == 'first'
    assert calls == 1
    assert history[1]['phase'] == 'publisher_prepare' and history[1]['error_type'] == 'ReadTimeout'
    assert history[1]['reason'] != 'source_unreachable'
    assert all(row['ended_at_unix'] >= row['started_at_unix'] for row in history)
    assert 'secret' not in str(history)
    events = [row for row in caplog.records if getattr(row, 'event', None) == 'vision_shield_stream_end']
    assert len(events) == 2
    assert events[0].task_id == 'job' and events[0].reason == 'source_eof'
    assert events[1].phase == 'publisher_prepare' and 'secret' not in caplog.text


async def test_attempt_history_is_available_without_exception_or_device_profile(tmp_path):
    from vision_shield.main import create_app

    history = [dict(attempt_id='ended', reason='source_eof', phase='source_read')]
    app = create_app(Settings(_env_file=None, media_root=tmp_path, debug=True))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.get = AsyncMock(return_value=dict(kind='stream', stream_attempts=history,
                                                         source_ref='rtsp://private:secret@input'))
        response = await client.get('/v1/tasks/00000000-0000-0000-0000-000000000001/diagnostics')
        assert response.status_code == 200
        assert response.json()['attempts'] == history
        assert 'secret' not in response.text


def test_failure_evidence_keeps_errno_and_location_without_exception_credentials():
    from vision_shield.media.attempts import failure_details

    try:
        raise ConnectionResetError(104, 'rtsp://private:secret@camera?token=hidden')
    except OSError as error:
        row = failure_details(error, 'source_read', debug=True)
    assert row['errno'] == 104 and row['phase'] == 'source_read'
    assert row['error_type'] == 'ConnectionResetError'
    assert row['at'] and 'secret' not in str(row) and 'hidden' not in str(row)


def test_ffmpeg_failure_retains_safe_protocol_reason_and_bounds_log():
    """保留 FFmpeg 的具体协议错误，但过滤地址、查询凭据与授权头。"""
    import av

    from vision_shield.media.attempts import failure_details

    error = av.error.InvalidDataError(1094995529, 'Invalid data found when processing input',
        filename='rtsp://private:camera-password@camera/live?token=hidden',
        log=(16, 'rtsp', 'method DESCRIBE failed: 401 Unauthorized\n'
             'rtsp://private:camera-password@camera/live?token=hidden '
             'token=hidden sign=publish-secret password="password-secret"\n'
             'Authorization: Basic credential-secret\n' + 'x' * 2000))
    row = failure_details(error, 'source_open', debug=True)
    assert row['ffmpeg_error'] == 'Invalid data found when processing input'
    assert 'DESCRIBE failed: 401 Unauthorized' in row['ffmpeg_log']
    assert len(row['ffmpeg_log']) <= 1024 and '\n' not in row['ffmpeg_log']
    assert not any(secret in str(row) for secret in (
        'camera-password', 'hidden', 'publish-secret', 'password-secret', 'credential-secret', 'rtsp://',
    ))


def test_real_ffmpeg_callback_is_kept_for_source_open_only(monkeypatch):
    """真实 PyAV 错误回调留下协议原因，源打开结束后恢复原日志级别。"""
    import av

    from vision_shield.media import streams

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_endpoint(url, mode=None, **kwargs):
        if mode == 'w':
            return Writer()
        av.logging.log(av.logging.ERROR, 'rtsp',
                       'method DESCRIBE failed: 401 Unauthorized rtsp://private:secret@camera')
        av.error.err_check(-1094995529)

    original_level = av.logging.get_level()
    monkeypatch.setattr(streams.av, 'open', open_endpoint)
    events = []
    with pytest.raises(av.error.InvalidDataError):
        streams.process_stream('rtsp://source', 'rtsp://output', object(),
                               Settings(_env_file=None, debug=True).model_dump(), threading.Event(), events.append)
    assert av.logging.get_level() == original_level
    ending = events[-1]['stream_end']
    assert 'DESCRIBE failed: 401 Unauthorized' in ending['ffmpeg_log']
    assert ending['phase'] == 'source_open' and 'secret' not in str(events)


@pytest.mark.parametrize('phase', ['source_open', 'source_read', 'source_decode', 'publish_open'])
def test_same_io_error_is_attributed_to_the_actual_endpoint_or_decoder(monkeypatch, phase):
    from vision_shield.media import streams

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    video = SimpleNamespace(type='video', codec_context=SimpleNamespace(name='h264'), width=64, height=48)

    class Reader:
        streams = SimpleNamespace(video=[video], audio=[])

        def demux(self):
            if phase == 'source_read':
                raise ConnectionResetError(104, 'secret')
            def decode():
                raise ConnectionResetError(104, 'secret')
            yield SimpleNamespace(stream=video, pts=None, decode=decode)

        def close(self):
            pass

    def open_endpoint(url, mode=None, **kwargs):
        if phase == ('publish_open' if mode == 'w' else 'source_open'):
            raise ConnectionResetError(104, 'rtsp://private:secret@camera')
        return Writer() if mode == 'w' else Reader()

    monkeypatch.setattr(streams.av, 'open', open_endpoint)
    events = []
    with pytest.raises(ConnectionResetError):
        streams.process_stream('rtsp://source', 'rtsp://output', object(),
                               Settings(_env_file=None, debug=True).model_dump(), threading.Event(), events.append)
    ending = events[-1]['stream_end']
    assert ending['phase'] == phase and ending['errno'] == 104
    assert ending['attempt_id'] == events[-1]['stats']['stream']['attempt_id']
    assert 'secret' not in str(events)


async def test_history_stays_bounded_and_marks_interrupted_owner_before_new_attempt(monkeypatch):
    from vision_shield.jobs import tasks

    history = [dict(run_id=str(i), generation=1, started_at_unix=1, ended_at_unix=2, reason='source_eof') for i in range(32)]
    history[-1].pop('ended_at_unix')
    record = dict(id='job', kind='stream', desired_state='run', generation=2, source_ref='rtsp://input', stream_attempts=history)
    store = SimpleNamespace(update_owned=AsyncMock(return_value=True), get=AsyncMock(return_value=record), finish=AsyncMock())
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtsp://output'), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    cancel = asyncio.Event()

    async def run(*args, **kwargs):
        origin = args[1]['config']['stream_started_monotonic']
        history = next(call.kwargs['stream_attempts'] for call in reversed(store.update_owned.call_args_list)
                       if 'stream_attempts' in call.kwargs)
        assert origin == history[-1]['started_monotonic']
        assert origin <= tasks.time.monotonic()
        cancel.set()
        return dict(returncode=-9, cancelled=True)

    monkeypatch.setattr(tasks, 'run_process', run)
    await tasks.run_stream(record, store, Settings(_env_file=None, debug=True), cancel, 'owner', 2,
                           cancellation=dict(reason='lease_lost', phase='worker_control'))
    result = store.update_owned.call_args.kwargs
    assert len(result['stream_attempts']) == 32 and result['stream_attempts_discarded'] == 1
    assert result['stream_attempts'][-2]['reason'] == 'worker_lost'
    assert result['stream_attempts'][-1]['reason'] == 'lease_lost'
    assert result['stream_attempts'][-1]['returncode'] == -9


async def test_fatal_child_failure_is_saved_before_terminal_state(monkeypatch):
    from vision_shield.jobs import tasks

    record = dict(id='job', kind='stream', desired_state='run', generation=1, source_ref='rtsp://input')
    saved = {}

    async def update(*args, **values):
        saved.update(copy.deepcopy(values))
        return True

    async def finish(*args, **kwargs):
        ending = saved['stream_attempts'][-1]
        assert ending['error_type'] == 'ValueError' and ending['phase'] == 'child_initialize'
        assert ending['ended_at_unix'] >= ending['started_at_unix']

    store = SimpleNamespace(update_owned=AsyncMock(side_effect=update), finish=AsyncMock(side_effect=finish))
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtsp://output'), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    monkeypatch.setattr(tasks, 'run_process', AsyncMock(return_value=dict(returncode=1, retryable=False,
        failure_type='ValueError', error=dict(code='processing_failed', message='Media processing failed'))))
    await tasks.run_stream(record, store, Settings(_env_file=None, debug=True), asyncio.Event(), 'owner', 1)
    store.finish.assert_awaited_once()


async def test_stop_during_model_recovery_does_not_start_another_attempt(monkeypatch):
    from vision_shield.jobs import tasks

    record = dict(id='job', kind='stream', desired_state='run', generation=1, source_ref='rtsp://input')
    cancel = asyncio.Event()
    async def update(*args, **values):
        if values.get('state') == 'recovering':
            record['desired_state'] = 'stopped'
            cancel.set()
        return True
    store = SimpleNamespace(update_owned=AsyncMock(side_effect=update), get=AsyncMock(return_value=record), finish=AsyncMock())
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtsp://output'), close=AsyncMock(return_value=True))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    run = AsyncMock(return_value=dict(returncode=1, retryable=True, failure_type='RuntimeError',
                                    error=dict(code='processing_failed', message='Model runtime failed')))
    monkeypatch.setattr(tasks, 'run_process', run)
    await tasks.run_stream(record, store, Settings(_env_file=None), cancel, 'owner', 1)
    run.assert_awaited_once()
    assert store.finish.call_args.args[3] == 'stopped'


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('debug', [False, True])
async def test_external_publisher_stop_persists_reason_before_releasing_ownership(monkeypatch, resident, debug):
    from vision_shield.jobs import tasks

    record = dict(id='job', kind='stream', desired_state='stopped', generation=1,
                  source_ref='rtsp://input', publish_url='rtmp://output')
    saved = {}
    terminal = False

    async def update(*args, **values):
        if terminal:
            return False
        saved.update(copy.deepcopy(values))
        return True

    async def close(*args, process_exited=False, media_closed=False):
        nonlocal terminal
        if process_exited or media_closed:
            assert media_closed == resident
            if debug:
                assert saved['stream_attempts'][-1]['reason'] == 'stop_requested'
            else:
                assert 'stream_attempts' not in saved
            terminal = True
        return True

    store = SimpleNamespace(update_owned=AsyncMock(side_effect=update), get=AsyncMock(return_value=record), finish=AsyncMock())
    publisher = SimpleNamespace(prepare=AsyncMock(return_value='rtmp://output'), close=AsyncMock(side_effect=close))
    monkeypatch.setattr(tasks, 'Publisher', lambda *args: publisher)
    monkeypatch.setattr(tasks, 'validate_source', AsyncMock(return_value=(httpx.URL(record['source_ref']), '127.0.0.1')))
    cancel = asyncio.Event()

    async def run(*args, **kwargs):
        assert kwargs.get('session') is session
        await args[3](dict(stream_end=dict(reason='stop_requested', phase='source_read', attempt_id='ended')))
        cancel.set()
        await kwargs['on_exit']()
        return dict(returncode=0, cancelled=True)

    monkeypatch.setattr(tasks, 'run_process', run)
    session = object() if resident else None
    await tasks.run_stream(record, store, Settings(_env_file=None, debug=debug), cancel, 'owner', 1,
                           cancellation=dict(reason='stop_requested', phase='worker_control'), session=session)
    assert terminal
    if debug:
        assert saved['stream_attempts'][-1]['attempt_id'] == 'ended'


def test_production_source_open_does_not_start_ffmpeg_log_capture(monkeypatch):
    from vision_shield.media import streams

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_endpoint(url, mode=None, **kwargs):
        if mode == 'w':
            return Writer()
        raise ConnectionResetError(104, 'private source')

    monkeypatch.setattr(streams.av, 'open', open_endpoint)
    monkeypatch.setattr(streams.av.logging, 'Capture', lambda: pytest.fail('No FFmpeg capture in production'))
    events = []
    with pytest.raises(ConnectionResetError):
        streams.process_stream('rtsp://source', 'rtsp://output', object(),
                               Settings(_env_file=None).model_dump(), threading.Event(), events.append)
    ending = events[-1]['stream_end']
    assert ending == dict(reason='source_open_error', phase='source_open', error_type='ConnectionResetError', errno=104)
    assert 'stream' not in events[-1]['stats']
