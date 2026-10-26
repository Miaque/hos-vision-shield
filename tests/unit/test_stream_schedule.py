"""Normal realtime jobs must use the adopted Route A schedule without hidden headers."""
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.config import Settings
from vision_shield.main import create_app


@pytest.mark.parametrize('headers,interval,tracking', [({}, 3, 'flow'),
    ({'X-Tracking-Mode': 'flow'}, 3, 'flow'),
    ({'X-Detection-Interval': '3'}, 3, 'flow'),
    ({'X-Detection-Interval': '1'}, 1, 'template'),
    ({'X-Detection-Interval': '3', 'X-Tracking-Mode': 'template'}, 3, 'template')])
async def test_stream_creation_persists_effective_schedule(tmp_path, monkeypatch, headers, interval, tracking):
    from vision_shield.api import routes

    monkeypatch.setattr(routes, 'require_worker', AsyncMock())
    monkeypatch.setattr(routes, 'validate_source', AsyncMock())
    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    record = dict(id='00000000-0000-0000-0000-000000000001', kind='stream', state='queued',
                  desired_state='run', ready=False, created_at='2026-10-26T00:00:00Z',
                  updated_at='2026-10-26T00:00:00Z', error=None,
                  stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.create = AsyncMock(return_value=record)
        response = await client.post('/v1/stream-tasks', headers=headers, json={'url': 'rtsp://camera/live'})
        assert response.status_code == 202
        options = app.state.store.create.call_args.kwargs
        assert options['detection_interval'] == interval and options['tracking'] == tracking


async def test_explicit_per_frame_stream_is_not_lost_when_persisting_defaults():
    from vision_shield.jobs.store import TaskStore

    store = TaskStore(AsyncMock())
    store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit: record)
    record = await store.create('stream', 'rtsp', 'rtsp://camera', detection_interval=1, tracking='template')
    assert record['detection_interval'] == 1 and record['tracking'] == 'template'
