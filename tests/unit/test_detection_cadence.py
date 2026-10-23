import numpy as np
import pytest
import httpx
from unittest.mock import AsyncMock

from vision_shield.redaction import FrameRedactor
from vision_shield.config import Settings
from vision_shield.main import create_app


class MovingTarget:
    def __init__(self):
        self.calls = []
        self.box = (20, 30, 44, 60)

    def detect(self, pixels):
        self.calls.append(pixels.shape)
        return [self.box] if self.box else []


def test_every_third_frame_keeps_template_tracking_and_all_output_frames():
    detector = MovingTarget()
    redactor = FrameRedactor(detector, detection_interval=3)
    texture = np.random.default_rng(76).integers(30, 130, (30, 24, 3), dtype=np.uint8)
    outputs = []
    for index in range(9):
        left = 20 + index * 6
        pixels = np.full((100, 180, 3), 180, np.uint8)
        pixels[30:60, left:left + 24] = texture
        detector.box = (left, 30, left + 24, 60)
        outputs.append(redactor.apply(pixels, index / 25))
        assert (outputs[-1][30:60, left:left + 24] == 64).all()
        assert np.array_equal(outputs[-1][:, :left - 6], pixels[:, :left - 6])
    assert len(outputs) == 9
    assert len(detector.calls) == 3
    assert redactor.detection_stats == dict(full_frames=3, skipped_frames=6, local_rechecks=0)


def test_new_target_on_skipped_frame_is_delayed_until_next_detection():
    detector = MovingTarget()
    detector.box = None
    redactor = FrameRedactor(detector, detection_interval=3)
    pixels = np.full((100, 180, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    detector.box = (20, 30, 44, 60)
    for index in (1, 2):
        assert np.array_equal(redactor.apply(pixels, index / 25), pixels)
    assert (redactor.apply(pixels, 3 / 25)[30:60, 20:44] == 64).all()
    assert redactor.detection_stats == dict(full_frames=2, skipped_frames=2, local_rechecks=0)


@pytest.mark.parametrize('interval', [0, 2, 4, True, 3.0])
def test_unsupported_detection_interval_is_rejected(interval):
    with pytest.raises(ValueError, match='detection interval'):
        FrameRedactor(MovingTarget(), detection_interval=interval)


@pytest.mark.parametrize('kind,interval,status', [('video', '3', 202), ('video', '1', 202),
                                               ('image', '3', 422), ('video', '2', 422)])
async def test_cadence_header_is_video_only_and_retains_private_counters(tmp_path, monkeypatch, kind, interval, status):
    from vision_shield.api import routes

    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    monkeypatch.setattr(routes, 'require_worker', AsyncMock())
    record = dict(id='00000000-0000-0000-0000-000000000001', kind=kind, state='queued', desired_state='run',
                  ready=False, created_at='2026-10-22T00:00:00Z', updated_at='2026-10-22T00:00:00Z', error=None,
                  stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0),
                  detection_stats=dict(full_frames=4, skipped_frames=8, local_rechecks=2))
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        app.state.store.create = AsyncMock(return_value=record)
        response = await client.post('/v1/file-tasks/uploads', headers={'X-Detection-Interval': interval},
                                     data={'kind': kind}, files={'file': ('input.mp4', b'video', 'video/mp4')})
        assert response.status_code == status
        if status == 202:
            assert app.state.store.create.call_args.kwargs['detection_interval'] == int(interval)
            assert 'detection_stats' not in response.json()
            app.state.store.get = AsyncMock(return_value=record)
            response = await client.get('/v1/tasks/' + record['id'])
            import json
            assert json.loads(response.headers['X-Detection-Counts']) == record['detection_stats']
        else:
            app.state.store.create.assert_not_called()


async def test_store_cadence_defaults_to_every_frame_and_rejects_nonvideo_or_invalid_intervals():
    from vision_shield.jobs.store import TaskStore

    store = TaskStore(AsyncMock())
    store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit: record)
    assert 'detection_interval' not in await store.create('video', 'upload', 'local')
    assert (await store.create('video', 'upload', 'local', detection_interval=3))['detection_interval'] == 3
    for kind, interval in [('image', 3), ('stream', 3), ('video', 0), ('video', True)]:
        with pytest.raises(ValueError):
            await store.create(kind, 'upload', 'local', detection_interval=interval)
