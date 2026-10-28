"""Whole-box LK is a gated last resort; existing successes and discovery win."""
from types import SimpleNamespace

import cv2
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
