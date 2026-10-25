import asyncio
import os
from pathlib import Path
import socket
import sys

from PIL import Image
import pytest

from vision_shield.media import runner
from vision_shield.config import Settings


COMMAND = [sys.executable, '-u', str(Path(__file__).parents[1] / 'fixtures/resident_probe.py'), '--persistent']


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16)).save(source)
    return dict(kind='image', source=str(source), directory=str(tmp_path / 'first'),
                config=dict(inference_device='cpu', max_image_pixels=1000))


async def run(session, payload, cancel=None, on_event=None, on_exit=None):
    events = []
    async def received(event):
        events.append(event)
        if on_event:
            await on_event(event)
    result = await asyncio.wait_for(runner.run_process(COMMAND[:-1], payload, cancel or asyncio.Event(),
        received, 0.2, session=session, on_exit=on_exit), 20)
    return result, events


async def test_consecutive_files_reuse_preflight_model_and_reset_task_state(payload, tmp_path):
    session = runner.MediaProcess(COMMAND)
    try:
        first, events = await run(session, payload)
        observed = next(event['observed'] for event in events if 'observed' in event)
        payload['directory'] = str(tmp_path / 'second')
        payload['config']['redaction_mode'] = 'mosaic'
        second, events = await run(session, payload)
        repeated = next(event['observed'] for event in events if 'observed' in event)
        assert first['returncode'] == second['returncode'] == 0
        assert observed['pid'] == repeated['pid'] and repeated['loads'] == 1
        assert observed['mode'] == 'solid' and repeated['mode'] == 'mosaic'
        assert repeated['previous_seconds'] == 0 and repeated['previous_candidates'] is None
        assert Path(first['result_ref']).read_bytes() == Path(second['result_ref']).read_bytes() == b'result'
    finally:
        await session.close()


async def test_stream_retry_and_source_switch_keep_model_and_confirm_closed_socket(payload):
    session = runner.MediaProcess(COMMAND)
    payload.update(kind='stream', source='retry', publish_url='rtmp://output')
    seen = []
    try:
        for source in ('retry', 'eof', 'another-source'):
            payload['source'] = source
            cancel = asyncio.Event()
            async def received(event):
                if 'publishing' in event:
                    seen.append(event['publishing'])
                    if source == 'another-source':
                        cancel.set()
            async def closed():
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', seen[-1]['port']))
            result, _ = await run(session, payload, cancel, received, closed)
            assert result['returncode'] == (1 if source == 'retry' else 0)
        assert len({event['pid'] for event in seen}) == 1
        assert all(event['loads'] == 1 for event in seen)
    finally:
        await session.close()


@pytest.mark.parametrize('source', ['stuck', 'fatal'])
async def test_timeout_or_fatal_error_reaps_child_before_callback_and_reloads(payload, tmp_path, source):
    session = runner.MediaProcess(COMMAND)
    stream = {**payload, 'kind': 'stream', 'source': source, 'publish_url': 'rtmp://output'}
    cancel, publishing, confirmed = asyncio.Event(), {}, []
    async def received(event):
        if 'publishing' in event:
            publishing.update(event['publishing'])
            if source == 'stuck':
                cancel.set()
    async def closed():
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', publishing['port']))
        confirmed.append(True)
    try:
        failed, _ = await run(session, stream, cancel, received, closed)
        assert failed['returncode'] != 0 and confirmed == [True]
        result, events = await run(session, payload)
        observed = next(event['observed'] for event in events if 'observed' in event)
        assert result['returncode'] == 0 and observed['pid'] != publishing['pid']
        assert observed['loads'] == 1
    finally:
        await session.close()


async def test_two_long_streams_and_two_files_have_independent_slots(payload, tmp_path):
    sessions = [runner.MediaProcess(COMMAND) for _ in range(4)]
    cancels = [asyncio.Event(), asyncio.Event()]
    ready = [asyncio.Event(), asyncio.Event()]
    publishing = [{}, {}]
    async def received(index, event):
        if 'publishing' in event:
            publishing[index].update(event['publishing'])
            ready[index].set()
    streams = [asyncio.create_task(run(sessions[index],
        {**payload, 'kind': 'stream', 'source': f'camera-{index}', 'publish_url': 'rtsp://output'},
        cancels[index], lambda event, index=index: received(index, event))) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 20)
        files = await asyncio.gather(*(run(sessions[index + 2],
            {**payload, 'directory': str(tmp_path / f'file-{index}')}) for index in range(2)))
        assert all(result['returncode'] == 0 for result, _ in files)
        assert all(not stream.done() for stream in streams)
        pids = {event['pid'] for event in publishing}
        pids.update(event['observed']['pid'] for _, events in files for event in events if 'observed' in event)
        assert len(pids) == 4
        cancels[0].set()
        await streams[0]
        result, events = await run(sessions[0], {**payload, 'kind': 'stream', 'source': 'eof', 'publish_url': 'rtsp://output'})
        assert next(event['publishing']['pid'] for event in events if 'publishing' in event) == publishing[0]['pid']
        assert not streams[1].done()
    finally:
        for cancel in cancels:
            cancel.set()
        await asyncio.gather(*streams, return_exceptions=True)
        await asyncio.gather(*(session.close() for session in sessions))


async def test_idle_resident_exits_on_parent_pipe_eof():
    session = runner.MediaProcess(COMMAND)
    try:
        await run(session, dict(kind='probe', config={}))
        process = session.process
        process.stdin.close()
        assert await asyncio.wait_for(process.wait(), 10) == 3
    finally:
        await session.close()


async def test_parent_event_failure_reclaims_active_publisher(payload):
    session = runner.MediaProcess(COMMAND)
    publishing, confirmed = {}, []
    async def received(event):
        if 'publishing' in event:
            publishing.update(event['publishing'])
            raise RuntimeError('control failure')
    async def closed():
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', publishing['port']))
        confirmed.append(True)
    with pytest.raises(RuntimeError, match='control failure'):
        await run(session, {**payload, 'kind': 'stream', 'source': 'camera', 'publish_url': 'rtmp://output'},
                  on_event=received, on_exit=closed)
    assert confirmed == [True] and session.process is None


@pytest.mark.parametrize('backend', ['cpu', pytest.param('quality_yolo', marks=pytest.mark.skipif(
    os.getenv('RUN_GPU_TESTS') != '1', reason='requires actual CUDA models'))])
async def test_real_preflight_and_consecutive_images_keep_same_model(payload, tmp_path, backend):
    command = [sys.executable, '-u', '-m', 'vision_shield.media.child', '--persistent']
    session = runner.MediaProcess(command)
    device = 'cpu' if backend == 'cpu' else 'cuda'
    payload['config'] = Settings(_env_file=None, detection_backend=backend, inference_device=device).model_dump(mode='json')
    Image.new('RGB', (320, 240)).save(payload['source'])
    try:
        first, events = await run(session, payload)
        pid = session.process.pid
        assert any(event.get('probe', {}).get('backend') == backend for event in events)
        assert first['returncode'] == 0 and first['model_reused']
        payload['directory'] = str(tmp_path / 'second')
        payload['config']['redaction_mode'] = 'mosaic'
        second, _ = await run(session, payload)
        assert second['returncode'] == 0 and second['model_reused'] and session.process.pid == pid
        for result in (first, second):
            with Image.open(result['result_ref']) as image:
                assert image.size == (320, 240)
    finally:
        await session.close()


async def test_model_settings_change_rebuilds_instead_of_reusing_stale_detector(payload):
    session = runner.MediaProcess(COMMAND)
    try:
        await run(session, payload)
        previous = session.process
        payload['config']['detection_max_side'] = 640
        result, _ = await run(session, payload)
        assert result['returncode'] == 0 and session.process.pid != previous.pid
        assert previous.returncode is not None
    finally:
        await session.close()


async def test_profiling_reclaims_resident_and_uses_isolated_child(payload):
    session = runner.MediaProcess(COMMAND)
    async def received(event):
        pass
    try:
        await run(session, payload)
        previous = session.process
        command = [sys.executable, '-u', '-c', 'import os,json; print(json.dumps(dict(pid=os.getpid())),flush=True)']
        result = await runner.run_process(command, {**payload, 'profile_detection': 'host'},
                                         asyncio.Event(), received, session=session)
        assert result['returncode'] == 0 and result['pid'] != previous.pid
        assert previous.returncode is not None and session.process is None
        result, _ = await run(session, payload)
        assert result['returncode'] == 0 and session.process.pid != previous.pid
    finally:
        await session.close()
