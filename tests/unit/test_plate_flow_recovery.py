"""Whole-box LK is a gated last resort; existing successes and discovery win."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import httpx
import numpy as np
import pytest

from vision_shield.detection import Detection
from vision_shield.redaction import FrameRedactor


@pytest.mark.parametrize('original_success', [True, False])
def test_existing_lk_and_template_successes_never_try_whole_box(monkeypatch, original_success):
    box = (60, 50, 100, 70)
    detector = SimpleNamespace(detect_regions=lambda pixels:[Detection('plate', box, .9)])
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', plate_flow_recovery=True, debug=True)
    redactor._plate_template_fallback = True

    def flow(gray, region, reference=None, *, whole_box=False):
        assert not whole_box
        return region if original_success else None

    monkeypatch.setattr(redactor, '_flow', flow)
    monkeypatch.setattr(redactor, '_follow', lambda gray, region, reference=None:region)
    pixels = np.full((160, 640, 3), 180, np.uint8)
    masks = [redactor.apply(pixels, moment) for moment in (0, .04, .08, .12)]
    assert redactor.detection_stats['full_frames'] == 2
    assert redactor.detection_stats['whole_flow_attempts'] == 0
    assert all((mask[50:70, 60:100] == 64).all() for mask in masks)


@pytest.mark.parametrize('case', ['recover', 'other_failed', 'new_proposal', 'competing', 'new_weak',
                                  'unknown_pending', 'expired', 'no_lookahead', 'disabled'])
def test_only_complete_safe_recovery_can_skip_a_joint_detection(monkeypatch, case):
    first, other, new = (60, 50, 100, 70), (400, 50, 440, 70), (540, 60, 580, 80)

    class Plates:
        calls = 0
        pending = ()

        def detect_regions(self, pixels):
            self.calls += 1
            self.unconfirmed_plate_boxes = self.pending
            self.has_unconfirmed_plates = bool(self.pending) or case == 'unknown_pending'
            return [Detection('plate', b, .9) for b in (first, other)]

        def plate_proposals(self, pixels):
            return [Detection('plate', new, .9)] if case == 'new_proposal' else []

    detector = Plates()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow',
                             plate_flow_recovery=case != 'disabled', debug=True)
    redactor._plate_template_fallback = case != 'no_lookahead'
    whole_calls = []

    def flow(gray, box, reference=None, *, whole_box=False):
        if whole_box:
            whole_calls.append(box)
            return None if box == other else (62, 50, 102, 70)
        return None if box == first or case == 'other_failed' else box

    monkeypatch.setattr(redactor, '_flow', flow)
    monkeypatch.setattr(redactor, '_follow', lambda *args:None)
    pixels = np.full((160, 640, 3), 180, np.uint8)
    if case == 'competing':
        detector.pending = ((80, 50, 110, 70),)
    elif case == 'new_weak':
        detector.pending = (new,)
    redactor.apply(pixels, 0)
    if case == 'expired':
        redactor.moment = .20
    redactor.apply(pixels, .24 if case == 'expired' else .04)
    should_attempt = case in ('recover', 'other_failed', 'new_proposal')
    assert bool(whole_calls) == should_attempt
    assert detector.calls == (1 if case == 'recover' else 2)
    if case == 'recover':
        assert redactor.history[0].seen == 0
        assert redactor.detection_stats['whole_flow_recoveries'] == 1
        assert redactor.detection_stats['full_rechecks'] == 0
    if case == 'new_proposal':
        assert redactor.detection_stats['proposal_rechecks'] == 1


def test_whole_box_still_requires_six_initial_points(monkeypatch):
    redactor = FrameRedactor(SimpleNamespace(), detection_interval=3, tracking='flow', plate_flow_recovery=True)
    gray = np.zeros((160, 220), np.uint8)
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', lambda *args, **kwargs:np.zeros((5, 1, 2), np.float32))

    def unexpected(*args, **kwargs):
        raise AssertionError('LK must not run with fewer than six points')

    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', unexpected)
    assert redactor._flow(gray, (60, 50, 100, 70), gray, whole_box=True) is None


@pytest.mark.parametrize('kind,schedule,value,status', [
    ('video', 'flow', '1', 202), ('stream', 'flow', '1', 202),
    ('image', 'flow', '1', 422), ('video', 'template', '1', 422),
    ('stream', 'template', '1', 422), ('video', 'flow', 'other', 422),
    ('video', 'flow', None, 202)])
async def test_candidate_is_explicit_per_task_and_default_stays_p0(tmp_path, monkeypatch, kind, schedule, value, status):
    from vision_shield.api import routes
    from vision_shield.config import Settings
    from vision_shield.main import create_app

    app = create_app(Settings(_env_file=None, media_root=tmp_path))
    monkeypatch.setattr(routes, 'require_worker', AsyncMock())
    monkeypatch.setattr(routes, 'validate_source', AsyncMock())
    record = dict(id='00000000-0000-0000-0000-000000000001', kind=kind, state='queued', desired_state='run',
        ready=False, created_at='2026-10-28T00:00:00Z', updated_at='2026-10-28T00:00:00Z', error=None,
        stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0))
    headers = {'X-Detection-Interval':'3', 'X-Tracking-Mode':schedule}
    if value is not None:
        headers['X-Plate-Flow-Recovery'] = value
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        app.state.store.create = AsyncMock(return_value=record)
        if kind == 'stream':
            response = await client.post('/v1/stream-tasks', headers=headers, json={'url':'rtsp://camera/live'})
        else:
            response = await client.post('/v1/file-tasks/uploads', headers=headers,
                data={'kind':kind}, files={'file':('input.mp4', b'video', 'video/mp4')})
        assert response.status_code == status
        if status == 202:
            options = app.state.store.create.call_args.kwargs
            assert options.get('plate_flow_recovery', False) == (value == '1')
        else:
            app.state.store.create.assert_not_called()


async def test_store_validates_and_preserves_candidate_switch():
    from vision_shield.jobs.store import TaskStore

    store = TaskStore(AsyncMock())
    store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit:record)
    assert 'plate_flow_recovery' not in await store.create('video', 'upload', 'local')
    record = await store.create('video', 'upload', 'local', detection_interval=3, tracking='flow', plate_flow_recovery=True)
    assert record['plate_flow_recovery'] is True
    for kind, interval, tracking, value in [('image', 1, 'template', True), ('video', 1, 'template', True),
                                           ('video', 3, 'flow', '1')]:
        with pytest.raises(ValueError, match='recovery'):
            await store.create(kind, 'upload', 'local', detection_interval=interval, tracking=tracking, plate_flow_recovery=value)
