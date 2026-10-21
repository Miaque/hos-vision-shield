from unittest.mock import AsyncMock
import httpx
import pytest
from vision_shield.config import Settings
from vision_shield.main import create_app
from vision_shield.jobs.store import TaskStore
from vision_shield.api import routes


@pytest.mark.parametrize('endpoint', ['file-tasks', 'stream-tasks', 'file-tasks/uploads'])
@pytest.mark.parametrize('mode', [None, 'solid', 'mosaic', 'blur'])
async def test_request_mode_is_validated_and_persisted(endpoint, mode, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, 'validate_source', AsyncMock())
    monkeypatch.setattr(routes, 'require_worker', AsyncMock())
    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        change = AsyncMock(side_effect=lambda task_id, operation, task, limit: task)
        app.state.store._change = change
        data = {} if mode is None else {'redaction_mode': mode}
        if endpoint.endswith('uploads'):
            response = await client.post('/v1/'+endpoint, data={'kind':'image', **data}, files={'file':('test.png', b'input')})
        else:
            data['url'] = 'rtsp://example.com/live' if endpoint=='stream-tasks' else 'https://example.com/a.png'
            if endpoint=='file-tasks': data['kind']='image'
            response = await client.post('/v1/'+endpoint, json=data)
        assert response.status_code == (422 if mode=='blur' else 202)
        if mode=='blur':
            change.assert_not_called()
        else:
            assert response.json()['redaction_mode'] == (mode or 'solid')
            assert change.call_args.args[2]['redaction_mode'] == (mode or 'solid')


async def test_store_rejects_invalid_mode_before_writing():
    store = TaskStore(AsyncMock())
    with pytest.raises(ValueError):
        await store.create('image','url','https://example.com/a.png',redaction_mode='blur')
    store.client.eval.assert_not_called()
