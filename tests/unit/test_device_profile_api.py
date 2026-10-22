import json
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.config import Settings
from vision_shield.main import create_app


def test_profile_header_and_sample_choices_are_described_in_openapi():
    paths = create_app(Settings(_env_file=None)).openapi()['paths']
    header = next(item for item in paths['/v1/file-tasks/uploads']['post']['parameters'] if item['name'] == 'X-Detection-Profile')
    assert header['in'] == 'header' and not header['required']
    sample = next(item for item in paths['/v1/tasks/{task_id}/diagnostics']['get']['parameters'] if item['name'] == 'sample')
    assert sample['schema']['enum'] == [64, 424, 768]


@pytest.mark.parametrize('kind,header,status', [('video', '1', 202), ('image', '1', 422),
                                               ('video', '0', 422), ('video', 'yes', 422)])
async def test_only_explicit_video_profile_request_reaches_worker(tmp_path, monkeypatch, kind, header, status):
    from vision_shield.api import routes

    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    monkeypatch.setattr(routes, 'require_worker', AsyncMock())
    record = dict(id='00000000-0000-0000-0000-000000000001', kind=kind, state='queued', desired_state='run',
                  ready=False, created_at='2026-10-22T00:00:00Z', updated_at='2026-10-22T00:00:00Z', error=None,
                  stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.create = AsyncMock(return_value=record)
        response = await client.post('/v1/file-tasks/uploads', headers={'X-Detection-Profile': header},
                                     data={'kind': kind}, files={'file': ('input.mp4', b'video', 'video/mp4')})
        assert response.status_code == status
        if status == 202:
            assert app.state.store.create.call_args.kwargs['profile_detection'] is True
            assert 'profile_detection' not in response.json()
        else:
            app.state.store.create.assert_not_called()


async def test_profile_summary_and_trace_are_scoped_to_owned_task_directory(tmp_path):
    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    task_id = '00000000-0000-0000-0000-000000000001'
    directory = tmp_path / task_id / 'attempt-1'
    directory.mkdir(parents=True)
    trace = directory / 'plate-profile-64.json'
    trace.write_text(json.dumps(dict(traceEvents=[])), 'utf-8')
    record = dict(id=task_id, generation=1, device_profile=dict(state='collecting', samples=[dict(frame=64)]))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.get = AsyncMock(return_value=record)
        base = f'/v1/tasks/{task_id}/diagnostics'
        assert (await client.get(base)).json() == record['device_profile']
        assert (await client.get(base + '?sample=64')).json() == dict(traceEvents=[])
        assert (await client.get(base + '?sample=65')).status_code == 422
        assert (await client.get(base + '?sample=424')).status_code == 409
        del record['device_profile']
        assert (await client.get(base)).status_code == 409
        app.state.store.get.return_value = None
        assert (await client.get(base)).status_code == 404


async def test_profile_request_is_private_store_metadata_and_video_only():
    from vision_shield.jobs.store import TaskStore

    store = TaskStore(AsyncMock())
    store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit: record)
    plain = await store.create('video', 'upload', 'local')
    assert 'profile_detection' not in plain
    profiled = await store.create('video', 'upload', 'local', profile_detection=True)
    assert profiled['profile_detection'] is True
    with pytest.raises(ValueError):
        await store.create('image', 'upload', 'local', profile_detection=True)


async def test_trace_resolved_outside_task_is_not_downloadable(tmp_path, monkeypatch):
    from pathlib import Path

    task_id = '00000000-0000-0000-0000-000000000001'
    outside = tmp_path / 'other-task.json'
    outside.write_text('{"private":true}', 'utf-8')
    resolve = Path.resolve
    def redirected(path, *args, **kwargs):
        return outside if path.name == 'plate-profile-64.json' else resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', redirected)
    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.get = AsyncMock(return_value=dict(generation=1, device_profile={}))
        response = await client.get(f'/v1/tasks/{task_id}/diagnostics?sample=64')
        assert response.status_code == 409 and 'private' not in response.text
