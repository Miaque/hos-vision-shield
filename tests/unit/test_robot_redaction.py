from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from vision_shield.config import Settings
from vision_shield.detection.egoblur import Detector
from vision_shield.media.video import masked_frame


def test_robot_detection_keeps_more_detail_and_expands_boxes():
    seen = []

    def predict(image):
        seen.append(tuple(image.shape))
        return torch.tensor([[100., 50., 200., 100.]]), None, torch.tensor([0.9]), None

    detector = Detector.__new__(Detector)
    detector.models = [predict]
    detector.max_side, detector.box_padding = 1280, 0.15
    boxes = detector.detect(np.zeros((1080, 3840, 3), dtype=np.uint8))
    assert seen == [(3, 360, 1280)]
    assert boxes[0] == pytest.approx((255, 127.5, 645, 322.5))


@pytest.mark.parametrize("values", [{"detection_max_side": 0}, {"detection_max_side": 4096},
                                    {"detection_box_padding": -0.1}, {"detection_box_padding": float("nan")}])
def test_detection_settings_are_bounded(values):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_short_dropout_is_masked_but_history_expires_and_resets():
    from vision_shield.redaction import FrameRedactor

    class IntermittentDetector:
        boxes = [(20, 20, 40, 40)]

        def detect(self, pixels):
            return self.boxes

    detector = IntermittentDetector()
    redactor = FrameRedactor(detector)

    def frame(pts, width=100, value=200):
        f = av.VideoFrame.from_ndarray(np.full((100, width, 3), value, dtype=np.uint8), format="rgb24")
        f.pts, f.time_base = pts, Fraction(1, 1000)
        return f

    masked_frame(frame(0), redactor)
    detector.boxes = []
    assert not masked_frame(frame(100), redactor).to_ndarray(format="rgb24")[30, 30].any()
    # Retained masks never refresh their own expiry.
    assert masked_frame(frame(300), redactor).to_ndarray(format="rgb24")[30, 30].min() == 200
    detector.boxes = [(20, 20, 40, 40)]
    masked_frame(frame(400), redactor)
    detector.boxes = []
    assert masked_frame(frame(450, width=120), redactor).to_ndarray(format="rgb24")[30, 30].min() == 200
    detector.boxes = [(20, 20, 40, 40)]
    masked_frame(frame(500), redactor)
    detector.boxes = []
    assert masked_frame(frame(550, value=50), redactor).to_ndarray(format="rgb24")[30, 30].min() == 50


def test_timing_is_preserved_with_short_dropout_compensation():
    from vision_shield.redaction import FrameRedactor

    class NoTargets:
        def detect(self, pixels):
            return []

    frame = av.VideoFrame.from_ndarray(np.full((100, 100, 3), 150, np.uint8), format="rgb24")
    frame.pts, frame.time_base = 120, Fraction(1, 100)
    output = masked_frame(frame, FrameRedactor(NoTargets()), Fraction(1))
    assert output.pts == 20 and output.time_base == Fraction(1, 100)


def test_history_is_bounded_and_clock_rewind_cannot_reuse_masks():
    from vision_shield.redaction import FrameRedactor

    class Targets:
        boxes = [(1, 1, 5, 5)]

        def detect(self, pixels):
            return self.boxes

    detector = Targets()
    redactor = FrameRedactor(detector)
    pixels = np.full((20, 20, 3), 100, np.uint8)
    for time in [0, 0.01, 0.02, 0.03]:
        redactor.apply(pixels, time)
    assert len(redactor.history) == 2
    detector.boxes = []
    assert redactor.apply(pixels, 0.01)[3, 3].min() == 100


def test_same_brightness_color_change_does_not_keep_old_masks():
    from vision_shield.redaction import FrameRedactor

    class Targets:
        boxes = [(1, 1, 5, 5)]

        def detect(self, pixels):
            return self.boxes

    detector = Targets()
    redactor = FrameRedactor(detector)
    red = np.full((20, 20, 3), [255, 0, 0], np.uint8)
    green = np.full((20, 20, 3), [0, 255, 0], np.uint8)
    redactor.apply(red, 0)
    detector.boxes = []
    assert np.array_equal(redactor.apply(green, 0.1)[3, 3], green[3, 3])
