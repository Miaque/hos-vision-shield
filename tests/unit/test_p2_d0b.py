"""The host observer may measure alternate LK but cannot change P0 state."""
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest

from vision_shield.detection import Detection
from vision_shield.detection import p2_d0b
from vision_shield.redaction import FrameRedactor


def run_shadow(enabled, *, report_padding=0):
    collector = SimpleNamespace(forward_counts=Counter(), report=lambda frames:dict(frames=frames, padding='x'*report_padding))

    class Plates:
        calls = 0
        has_unconfirmed_plates = True
        unconfirmed_plate_boxes = ((200, 120, 210, 128),)

        def detect_regions(self, pixels, kind=None):
            self.calls += 1
            collector.forward_counts['primary-plate'] += 1
            return [Detection('plate', (60, 50, 80, 60), 1)]

    pixels = np.random.default_rng(52).integers(20, 200, (160, 240, 3), np.uint8)
    saved = pixels.copy()
    detector = Plates()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    redactor._plate_template_fallback = True
    with ExitStack() as stack:
        if enabled:
            p2_d0b.install(stack, collector)
        masks = [redactor.apply(pixels, index*.04) for index in range(3)]
        report = collector.report(3)
    np.testing.assert_array_equal(pixels, saved)
    assert sys.getprofile() is None
    return masks, [(t.seen, t.checked, t.box, t.kind) for t in redactor.history], redactor.detection_stats, detector.calls, report


def test_observer_preserves_masks_lifetimes_gates_and_model_call_count():
    baseline, observed = run_shadow(False), run_shadow(True)
    for before, after in zip(baseline[0], observed[0], strict=True):
        np.testing.assert_array_equal(before, after)
    assert baseline[1:4] == observed[1:4]
    evidence = observed[4]['p2_d0b']
    assert evidence['rows'] and not evidence['truncated']
    assert evidence['rows'][0]['targets'][0]['original_blocked']
    assert evidence['rows'][0]['actual_forwards'] == {'primary-plate': 1}


def test_observer_bounds_the_child_pipe_addition():
    report = run_shadow(True, report_padding=200*1024)[4]['p2_d0b']
    assert report['truncated'] and report['rows'] == []


def test_profile_hook_is_restored_on_optical_flow_error(monkeypatch):
    _, whole = p2_d0b.shadow_function(FrameRedactor)
    owner = FrameRedactor(SimpleNamespace(), detection_interval=3, tracking='flow')
    pixels = np.zeros((80, 100), np.uint8)

    def failed(*args, **kwargs):
        raise RuntimeError('corner failure')

    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', failed)
    with pytest.raises(RuntimeError, match='corner failure'):
        p2_d0b.observe(whole, owner, pixels, (20, 20, 50, 40), pixels)
    assert sys.getprofile() is None
