import cv2
import numpy as np
import pytest
from unittest.mock import AsyncMock
import httpx

from vision_shield.redaction import FrameRedactor
from tests.unit.test_detection_cadence import MovingTarget
from vision_shield.config import Settings
from vision_shield.main import create_app
from vision_shield.detection import Detection


def texture():
    return cv2.GaussianBlur(np.random.default_rng(91).integers(20, 220, (160, 220), np.uint8), (3, 3), 0)


def test_sparse_flow_follows_translation_and_scale_with_stationary_background():
    previous = np.full((160, 220), 180, np.uint8)
    previous[50:100, 60:120] = texture()[50:100, 60:120]
    matrix = cv2.getRotationMatrix2D((90, 75), 0, 1.12)
    matrix[:, 2] += (8, 4)
    current = cv2.warpAffine(previous, matrix, (220, 160), borderValue=180)
    redactor = FrameRedactor(MovingTarget(), detection_interval=3, tracking='flow', debug=True)
    redactor.gray = previous
    box = redactor._flow(current, (60, 50, 120, 100))
    expected = [64.4, 51, 131.6, 107]
    np.testing.assert_allclose(box, expected, atol=1.2, rtol=0)


@pytest.mark.parametrize('current', ['blank', 'unrelated'])
def test_flow_rejects_occlusion_or_inconsistent_points(current):
    redactor = FrameRedactor(MovingTarget(), detection_interval=3, tracking='flow', debug=True)
    redactor.gray = texture()
    gray = np.full((160, 220), 180, np.uint8) if current == 'blank' else np.random.default_rng(92).integers(0, 255, (160, 220), np.uint8)
    assert redactor._flow(gray, (60, 50, 120, 100)) is None


def test_failed_flow_forces_full_detection_on_the_skipped_frame():
    detector = MovingTarget()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    pixels = np.full((100, 180, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    detector.box = (70, 30, 94, 60)
    output = redactor.apply(pixels, 0.04)
    assert (output[30:60, 70:94] == 64).all()
    assert redactor.detection_stats['full_frames'] == 2
    assert redactor.detection_stats['full_rechecks'] == 1
    assert redactor.detection_stats['flow_failures'] == 1


def test_scene_reset_forces_detection_instead_of_publishing_an_unmasked_skip():
    redactor = FrameRedactor(MovingTarget(), detection_interval=3, tracking='flow', debug=True)
    redactor.apply(np.full((100, 180, 3), 80, np.uint8), 0)
    output = redactor.apply(np.full((100, 180, 3), 180, np.uint8), 0.04)
    assert (output[30:60, 20:44] == 64).all()
    assert redactor.detection_stats['full_frames'] == 2 and redactor.detection_stats['full_rechecks'] == 1


@pytest.mark.parametrize('age,template_valid', [(.08, False), (.28, True), (.28, False)])
def test_stale_plate_flow_rechecks_when_original_template_rejects_background(monkeypatch, age, template_valid):
    class Plates:
        visible = True

        def detect_regions(self, pixels):
            return [Detection('plate', (60, 50, 120, 80), .9)] if self.visible else []

        def detect_faces(self, pixels):
            return []

    detector = Plates()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    monkeypatch.setattr(redactor, '_flow', lambda gray, box, reference=None: box)
    monkeypatch.setattr(redactor, '_follow', lambda gray, box, reference=None: box if template_valid else None)
    monkeypatch.setattr(redactor, '_recheck', lambda *a: None)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    detector.visible = False
    redactor.apply(pixels, age / 2)
    actual = redactor.apply(pixels, age)
    expired = age > .2 and not template_valid
    assert redactor.detection_stats['full_rechecks'] == int(expired)
    assert bool((actual[50:80, 60:120] == 64).all()) is not expired


def test_good_flow_skips_inference_but_does_not_renew_detection_lifetime():
    detector = MovingTarget()
    detector.box = (60, 50, 120, 100)
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    pixels = np.repeat(texture()[:, :, None], 3, axis=2)
    redactor.apply(pixels, 0)
    output = redactor.apply(pixels, 0.04)
    assert (output[51:99, 61:119] == 64).all()
    assert len(detector.calls) == 1
    assert redactor.history[0].seen == 0
    assert redactor.detection_stats['flow_updates'] == 1
    assert redactor.detection_stats['skipped_frames'] == 1


def test_forward_backward_inconsistency_cannot_be_accepted_as_motion(monkeypatch):
    redactor = FrameRedactor(MovingTarget(), detection_interval=3, tracking='flow', debug=True)
    redactor.gray = texture()
    calls = 0

    def bad_reverse(previous, current, points, following, **kwargs):
        nonlocal calls
        calls += 1
        return points + (0 if calls == 1 else 10), np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', bad_reverse)
    assert redactor._flow(redactor.gray, (60, 50, 120, 100)) is None


@pytest.mark.parametrize('first,current', [(None, (60, 50, 120, 100)),
                                          ((60, 0, 120, 28), (50, 0, 125, 60))])
def test_face_recheck_covers_new_or_partially_entering_target(monkeypatch, first, current):
    class Proposals:
        box = first

        def detect_regions(self, pixels, **kwargs):
            return [Detection('face', self.box, .8)] if self.box else []

        detect_faces = detect_regions

    detector = Proposals()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    monkeypatch.setattr(redactor, '_flow', lambda gray, box, reference=None: box)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    detector.box = current
    actual = redactor.apply(pixels, .04)
    x,y,a,b = current
    assert (actual[y:b, x:a] == 64).all()
    assert redactor.detection_stats['face_rechecks'] == 1


def test_full_frame_face_detection_renews_lifetime_on_each_frame(monkeypatch):
    class Proposals:
        def detect_regions(self, pixels, **kwargs):
            return [Detection('face', (60, 50, 120, 100), .8)]

        detect_faces = detect_regions

    redactor = FrameRedactor(Proposals(), detection_interval=3, tracking='flow', debug=True)
    monkeypatch.setattr(redactor, '_flow', lambda gray, box, reference=None: box)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    redactor.apply(pixels, .04)
    assert redactor.history[0].seen == .04
    redactor.apply(pixels, .08)
    assert redactor.history[0].seen == .08
    assert redactor.detection_stats['face_rechecks'] == 2


def test_unconfirmed_proposal_never_creates_a_mask():
    class Proposals:
        def detect_regions(self, pixels, **kwargs):
            return []

        def plate_proposals(self, pixels):
            return [Detection('plate', (60, 50, 120, 100), .4)]

    redactor = FrameRedactor(Proposals(), detection_interval=3, tracking='flow', debug=True)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    np.testing.assert_array_equal(redactor.apply(pixels, .04), pixels)
    assert redactor.detection_stats['proposal_rechecks'] == 1


def test_proposal_inference_is_included_in_the_existing_inference_time(monkeypatch):
    from vision_shield import redaction

    clock = [0.0]
    monkeypatch.setattr(redaction.time, 'monotonic', lambda: clock[0])

    class Proposals:
        def detect_regions(self, pixels):
            return []

        def plate_proposals(self, pixels):
            clock[0] += .01
            return []

    redactor = FrameRedactor(Proposals(), detection_interval=3, tracking='flow', debug=True)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    redactor.apply(pixels, .04)
    assert redactor.inference_seconds == pytest.approx(.01)


@pytest.mark.parametrize('uncovered_plate', [False, True])
def test_face_proposal_rechecks_only_faces_unless_a_plate_also_needs_confirmation(monkeypatch, uncovered_plate):
    class Proposals:
        global_calls = face_calls = 0
        face = False

        def detect_regions(self, pixels):
            self.global_calls += 1
            return self.proposals(pixels)

        def detect_faces(self, pixels):
            self.face_calls += 1
            return [c for c in self.proposals(pixels) if c.kind == 'face']

        def proposals(self, pixels):
            return ([Detection('face', (60, 50, 120, 100), .8)] if self.face else []) + (
                [Detection('plate', (140, 50, 200, 75), .8)] if self.face and uncovered_plate else [])

        def plate_proposals(self, pixels):
            return [c for c in self.proposals(pixels) if c.kind == 'plate']

    detector = Proposals()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    detector.face = True
    output = redactor.apply(pixels, .04)
    assert (output[50:100, 60:120] == 64).all()
    assert detector.global_calls == (2 if uncovered_plate else 1)
    assert detector.face_calls == (0 if uncovered_plate else 1)
    assert redactor.detection_stats['face_rechecks'] == (0 if uncovered_plate else 1)
    assert redactor.detection_stats['full_frames'] == (2 if uncovered_plate else 1)
    assert redactor.detection_stats['full_rechecks'] == int(uncovered_plate)
    if uncovered_plate:
        assert (output[50:75, 140:200] == 64).all()


def test_auxiliary_only_face_extent_is_checked_on_every_non_joint_frame(monkeypatch):
    class PartialPrimary:
        def detect_regions(self, pixels):
            return [Detection('face', (60, 70, 100, 105), .8)]

        proposals = detect_regions

        def plate_proposals(self, pixels):
            return []

        def detect_faces(self, pixels):
            return [Detection('face', (60, 50, 100, 105), .8)]

    redactor = FrameRedactor(PartialPrimary(), detection_interval=3, tracking='flow', debug=True)
    monkeypatch.setattr(redactor, '_flow', lambda gray, box, reference=None: box)
    pixels = np.full((160, 220, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    assert (redactor.apply(pixels, .04)[50:105, 60:100] == 64).all()
    assert redactor.detection_stats['face_rechecks'] == 1


@pytest.mark.parametrize('tracking,interval', [('bad', 3), ('flow', 1)])
def test_flow_option_rejects_invalid_combinations(tracking, interval):
    with pytest.raises(ValueError, match='tracking'):
        FrameRedactor(MovingTarget(), detection_interval=interval, tracking=tracking)


@pytest.mark.parametrize('kind,interval,tracking,status', [('video', '3', 'flow', 202),
    ('video', '3', 'template', 202), ('video', '1', 'flow', 422), ('video', None, 'flow', 422),
    ('image', '3', 'flow', 422), ('video', '3', 'invalid', 422)])
async def test_flow_request_is_explicit_and_restricted_to_third_frame_video(tmp_path, monkeypatch, kind, interval, tracking, status):
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
        headers = {'X-Tracking-Mode': tracking}
        if interval is not None:
            headers['X-Detection-Interval'] = interval
        response = await client.post('/v1/file-tasks/uploads', headers=headers,
                                     data={'kind': kind}, files={'file': ('input.mp4', b'video', 'video/mp4')})
        assert response.status_code == status
        if status == 202:
            assert app.state.store.create.call_args.kwargs['tracking'] == tracking
        else:
            app.state.store.create.assert_not_called()


async def test_store_rejects_flow_outside_video_or_stream_interval_three():
    from vision_shield.jobs.store import TaskStore

    store = TaskStore(AsyncMock())
    store._change = AsyncMock(side_effect=lambda task_id, operation, record, limit: record)
    assert 'tracking' not in await store.create('video', 'upload', 'local')
    assert (await store.create('video', 'upload', 'local', detection_interval=3, tracking='flow'))['tracking'] == 'flow'
    assert (await store.create('stream', 'rtsp', 'rtsp://camera', detection_interval=3, tracking='flow'))['tracking'] == 'flow'
    for kind, interval in [('video', 1), ('image', 3), ('stream', 1)]:
        with pytest.raises(ValueError):
            await store.create(kind, 'upload', 'local', detection_interval=interval, tracking='flow')
