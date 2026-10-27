"""Explain tracking failures without changing tracking decisions."""

from collections import deque
from contextlib import ExitStack
import sys

import cv2
import numpy as np
import pytest

from vision_shield.detection.flow_profile import FlowProfile
from vision_shield.detection import Detection
from vision_shield.redaction import FrameRedactor, _Track


def scenario(flow, *, tiny=False, broken=False):
    class Detector:
        has_unconfirmed_plates = False

        def detect_faces(self, pixels):
            return []

        def detect_regions(self, pixels, *, kind=None):
            if broken:
                raise RuntimeError('inference failed')
            for family in ('primary', 'detail'):
                for category in ('face', 'plate') if kind is None else (kind,):
                    flow.forward(f'forward-{family}-{category}')
            return []

    pixels = np.full((140, 260, 3), 180, np.uint8)
    texture = cv2.GaussianBlur(np.random.default_rng(91).integers(20, 220, (40, 80), np.uint8), (3, 3), 0)
    pixels[60:100, 140:220] = texture[:, :, None]
    redactor = FrameRedactor(Detector(), detection_interval=3, tracking='flow', debug=True)
    redactor._plate_template_fallback = True
    redactor.shape, redactor.moment, redactor.gray = pixels.shape, 0., pixels[:, :, 0]
    rows, cols = np.linspace(0,139,36,dtype=int), np.linspace(0,259,64,dtype=int)
    redactor.previous = pixels[rows[:,None],cols[None,:]].astype(np.float32)
    failed = (20, 60, 24, 64) if tiny else (20, 60, 60, 80)
    redactor.history = deque([_Track(0., 0., failed, 'plate'), _Track(0., 0., (140, 60, 220, 100), 'plate')], maxlen=128)
    redactor.frame_index = 1
    return redactor, pixels


@pytest.mark.parametrize('tiny,reason', [(False, 'not_enough_features'), (True, 'patch_too_small')])
def test_one_reference_only_failure_escalates_full_frame_with_healthy_sibling(tiny, reason):
    flow = FlowProfile()
    plain, pixels = scenario(flow, tiny=tiny)
    expected = plain.regions(pixels, .04)
    observed, pixels = scenario(flow, tiny=tiny)
    previous = sys.gettrace()
    with ExitStack() as stack:
        flow.install(stack)
        actual = observed.regions(pixels, .04)
    assert sys.gettrace() is previous
    assert actual == expected and observed.detection_stats == plain.detection_stats
    report = flow.report()
    assert report['raw_failure_reasons'] == {reason: 1}
    assert report['additional_full_frames'] == {'track_failure': 1}
    assert report['failure_reason_full_frames'] == {reason: 1}
    assert report['single_failed_plate_with_healthy_sibling_frames'] == 1
    assert report['track_failure_full_model_calls']['forward-primary-face'] == 1
    assert report['track_failure_full_model_calls']['forward-detail-face'] == 1
    assert report['samples'][0]['reason'] == reason
    assert report['samples'][0]['scheduled'] is False


def test_tracing_restores_previous_hook_and_propagates_inference_error():
    flow = FlowProfile()
    redactor, pixels = scenario(flow, broken=True)
    previous = sys.gettrace()
    with ExitStack() as stack:
        flow.install(stack)
        with pytest.raises(RuntimeError, match='inference failed'):
            redactor.regions(pixels, .04)
    assert sys.gettrace() is previous
    assert flow.report()['additional_full_frames'] == {}


def test_failure_samples_are_bounded_and_aggregate_counts_remain_complete():
    flow = FlowProfile()
    with ExitStack() as stack:
        flow.install(stack)
        for _ in range(5):
            redactor, pixels = scenario(flow)
            redactor.regions(pixels, .04)
    report = flow.report()
    assert report['raw_failure_reasons'] == {'not_enough_features': 5}
    assert report['failure_reason_full_frames'] == {'not_enough_features': 5}
    assert len(report['samples']) == 2


@pytest.mark.parametrize('trigger', ['scheduled', 'scene_reset', 'new_proposal'])
def test_full_frame_triggers_are_counted_separately(trigger):
    flow = FlowProfile()
    redactor, pixels = scenario(flow)
    pixels[60:80, 20:60] = pixels[60:80, 140:180]
    moment = .04
    if trigger == 'scheduled':
        redactor.frame_index = 0
    elif trigger == 'scene_reset':
        moment = .3
    else:
        redactor.detector.plate_proposals = lambda pixels: [Detection('plate', (20, 20, 60, 40), .9)]
    with ExitStack() as stack:
        flow.install(stack)
        redactor.regions(pixels, moment)
    report = flow.report()
    assert report['additional_full_frames'] == ({} if trigger == 'scheduled' else {trigger: 1})
    assert report['failure_reason_full_frames'] == {}


@pytest.mark.parametrize('unconfirmed', [False, True])
def test_template_rescue_and_global_unconfirmed_gate_are_distinguished(monkeypatch, unconfirmed):
    flow = FlowProfile()
    redactor, pixels = scenario(flow)
    pixels[60:80, 20:60] = pixels[60:80, 140:180]
    redactor.detector.has_unconfirmed_plates = unconfirmed
    monkeypatch.setattr(cv2, 'goodFeaturesToTrack', lambda *a, **kw: None)
    with ExitStack() as stack:
        flow.install(stack)
        redactor.regions(pixels, .04)
    report = flow.report()
    assert report['raw_failure_reasons'] == {'not_enough_features': 2}
    assert report.get('template_rescues', 0) == (0 if unconfirmed else 2)
    assert report.get('fallback_blocked_by_unconfirmed_plate', 0) == (2 if unconfirmed else 0)
    assert report['additional_full_frames'] == ({'track_failure': 1} if unconfirmed else {})


def test_good_flow_rejected_by_stale_template_is_not_a_raw_lk_failure(monkeypatch):
    flow = FlowProfile()
    redactor, pixels = scenario(flow)
    pixels[60:80, 20:60] = pixels[60:80, 140:180]
    redactor.moment = .24
    monkeypatch.setattr(redactor, '_follow', lambda *a, **kw: None)
    with ExitStack() as stack:
        flow.install(stack)
        redactor.regions(pixels, .28)
    report = flow.report()
    assert report['raw_failure_reasons'] == {}
    assert report['stale_template_rejections'] == 2
    assert report['failure_reason_full_frames'] == {'stale_template_rejected': 1}
