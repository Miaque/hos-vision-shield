from types import SimpleNamespace

import numpy as np
import pytest

from vision_shield.detection import Detection
from vision_shield.redaction import FrameRedactor


FIRST = (40, 40, 80, 60)
RECOVERED = (46, 42, 86, 62)
OTHER = (360, 40, 400, 60)
NEW = (500, 50, 540, 70)
FACE = (200, 100, 230, 140)


@pytest.fixture
def plate_recovery(monkeypatch):
    class Plates:
        pending = ()
        faces = []
        proposals = []
        visible = [Detection('plate', FIRST, .9), Detection('plate', OTHER, .9)]

        def detect_regions(self, pixels, *, kind=None):
            calls.append(('full' if kind is None else kind, pixels.shape))
            if kind is None:
                self.unconfirmed_plate_boxes = self.pending
                self.has_unconfirmed_plates = bool(self.pending)
            return self.visible + self.faces

        def detect_faces(self, pixels):
            calls.append(('faces', pixels.shape))
            return self.faces

        def plate_proposals(self, pixels):
            calls.append(('proposals', pixels.shape))
            return self.proposals

        def plate_needs_work(self, pixels, box):
            return True

    calls, follows, local = [], [], []
    detector = Plates()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    redactor._plate_template_fallback = True
    state = SimpleNamespace(detector=detector, redactor=redactor, calls=calls, follows=follows,
                            local=local, roi=RECOVERED, fail_all=False,
                            pixels=np.full((180, 640, 3), 180, np.uint8))

    def flow(gray, box, reference=None):
        return None if box == FIRST or state.fail_all else box

    def follow(gray, box, reference=None):
        follows.append(box)
        return None if box == FIRST or state.fail_all else box

    def recheck(pixels, box, kind='', gray=None):
        local.append((box, kind))
        if isinstance(state.roi, Exception):
            raise state.roi
        return state.roi if box == FIRST else None

    monkeypatch.setattr(redactor, '_flow', flow)
    monkeypatch.setattr(redactor, '_follow', follow)
    monkeypatch.setattr(redactor, '_recheck', recheck)
    redactor.apply(state.pixels, 0)
    calls.clear()
    follows.clear()
    local.clear()
    return state


def test_single_plate_roi_recovery_keeps_discovery_faces_and_confirmed_lifetime(plate_recovery):
    state = plate_recovery
    state.detector.faces = [Detection('face', FACE, .9)]
    output = state.redactor.apply(state.pixels, .04)
    operations = [name for name, shape in state.calls]
    assert 'full' not in operations
    assert operations.count('proposals') == operations.count('faces') == 1
    assert state.local == [(FIRST, 'plate')]
    assert OTHER not in state.follows
    assert state.redactor.detection_stats['full_frames'] == 1
    assert state.redactor.detection_stats['full_rechecks'] == 0
    recovered = [track for track in state.redactor.history
                 if track.kind == 'plate' and state.redactor._overlap(track.box, RECOVERED) >= .2]
    assert len(recovered) == 1 and recovered[0].seen == recovered[0].checked == .04
    for x, y, a, b in (RECOVERED, OTHER, FACE):
        assert (output[y:b, x:a] == 64).all()


def test_new_plate_proposal_still_gets_full_quality_detection(plate_recovery):
    state = plate_recovery
    state.detector.proposals = [Detection('plate', NEW, .4)]
    state.detector.visible = state.detector.visible + [Detection('plate', NEW, .9)]
    output = state.redactor.apply(state.pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == 1
    assert (output[50:70, 500:540] == 64).all()
    assert any(track.kind == 'plate' and track.box == NEW and track.seen == .04
               for track in state.redactor.history)
    assert all(box != OTHER for box, kind in state.local)


def test_full_escalation_keeps_already_confirmed_roi_edges(plate_recovery):
    state = plate_recovery
    state.detector.proposals = [Detection('plate', NEW, .4)]
    state.detector.visible = state.detector.visible + [Detection('plate', NEW, .9)]
    output = state.redactor.apply(state.pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == 1
    assert state.local == [(FIRST, 'plate')]
    for x, y, a, b in (RECOVERED, NEW):
        assert (output[y:b, x:a] == 64).all()


def test_full_escalation_keeps_other_raw_flow_when_full_and_template_miss_it(plate_recovery, monkeypatch):
    state = plate_recovery
    state.detector.proposals = [Detection('plate', NEW, .4)]
    state.detector.visible = [Detection('plate', FIRST, .9), Detection('plate', NEW, .9)]

    def reject(gray, box, reference=None):
        state.follows.append(box)
        return None

    monkeypatch.setattr(state.redactor, '_follow', reject)
    output = state.redactor.apply(state.pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == 1
    assert state.local == [(FIRST, 'plate')]
    assert OTHER not in state.follows
    assert (output[40:60, 360:400] == 64).all()


@pytest.mark.parametrize('candidate,covered', [
    ((48, 44, 76, 58), True),
    ((40, 40, 50, 60), False),
    ((362, 42, 398, 58), True),
    (None, False),
])
def test_pending_plate_requires_current_confirmed_coverage_before_skipping_full(plate_recovery, candidate, covered):
    state = plate_recovery
    state.detector.has_unconfirmed_plates = True
    state.detector.unconfirmed_plate_boxes = (candidate,) if candidate is not None else None
    output = state.redactor.apply(state.pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == int(not covered)
    if covered:
        assert state.local == [(FIRST, 'plate')]
        assert OTHER not in state.follows
        assert any(track.kind == 'plate' and track.seen == .04
                   and state.redactor._overlap(track.box, RECOVERED) >= .2
                   for track in state.redactor.history)
    if candidate is not None:
        x, y, a, b = candidate
        assert (output[y:b, x:a] == 64).all()


@pytest.mark.parametrize('reason', ['new_pending', 'stale_snapshot', 'expired_track',
                                   'global_failure', 'scene_reset', 'periodic'])
def test_roi_does_not_replace_required_full_detection(plate_recovery, reason):
    state = plate_recovery
    pixels = state.pixels
    if reason == 'new_pending':
        state.detector.has_unconfirmed_plates = True
        state.detector.unconfirmed_plate_boxes = (NEW,)
    elif reason == 'stale_snapshot':
        state.redactor._last_joint_detection = -.3
    elif reason == 'expired_track':
        state.redactor.history[0] = state.redactor.history[0]._replace(seen=-.3)
    elif reason == 'global_failure':
        state.fail_all = True
    elif reason == 'scene_reset':
        pixels = np.full_like(pixels, 70)
    elif reason == 'periodic':
        state.redactor.frame_index = 3
    state.redactor.apply(pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == 1
    assert state.redactor.detection_stats['full_frames'] == 2


def test_failed_roi_escalates_without_repeating_the_same_target_roi(plate_recovery):
    state = plate_recovery
    state.roi = None
    state.detector.visible = [Detection('plate', OTHER, .9)]
    output = state.redactor.apply(state.pixels, .04)
    assert sum(name == 'full' for name, shape in state.calls) == 1
    assert state.local == [(FIRST, 'plate')]
    assert OTHER not in state.follows
    assert (output[40:60, 40:80] == 64).all()


def test_roi_model_failure_cannot_publish_a_partial_frame(plate_recovery):
    state = plate_recovery
    state.roi = RuntimeError('plate model unavailable')
    with pytest.raises(RuntimeError, match='plate model unavailable'):
        state.redactor.apply(state.pixels, .04)
