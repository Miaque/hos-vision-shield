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
    detector.face_threshold = 0.1
    detector.device = torch.device("cpu")
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
    assert (masked_frame(frame(100), redactor).to_ndarray(format="rgb24")[30, 30] == 64).all()
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
    assert len(redactor.history) == 1, "连续确认的同一目标不应重复堆积历史框"
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


@pytest.mark.parametrize("local_confirmation", [False, True])
def test_moving_confirmed_mask_follows_pixels_and_only_real_detections_extend_expiry(local_confirmation):
    from vision_shield.redaction import FrameRedactor

    texture = np.random.default_rng(7).integers(40, 200, (20, 20, 3), dtype=np.uint8)
    texture[1, 1] = [255, 0, 0]
    calls = []
    class MissingGlobal:
        first = True

        def detect(self, pixels):
            calls.append(pixels.shape)
            if pixels.shape == (100, 200, 3):
                if self.first:
                    self.first = False
                    return [(20, 30, 40, 50)]
                return []
            if not local_confirmation:
                return []
            # 已知夹具目标映射到放大的局部视图，不依赖插值后仍有精确 RGB 标记。
            crop_left, crop_right = max(0, left - 38), min(200, left + 58)
            xs, ys = pixels.shape[1] / (crop_right - crop_left), pixels.shape[0] / 88
            return [((left - crop_left) * xs, 30 * ys, (left + 20 - crop_left) * xs, 50 * ys)]

    redactor = FrameRedactor(MissingGlobal())
    for index in range(12):
        left = 20 + index * 7
        pixels = np.full((100, 200, 3), 170, np.uint8)
        pixels[30:50, left:left + 20] = texture
        original = pixels.copy()
        actual = redactor.apply(pixels, index / 20)
        if index <= 9 or local_confirmation:
            assert (actual[31:49, left + 1:left + 19] == 64).all(), "短暂漏检时遮挡必须跟随目标，不能留在旧位置"
        if index == 11 and not local_confirmation:
            assert np.array_equal(actual, original), "仅跟踪成功不能无限续期"
        assert np.array_equal(pixels, original)
    assert sum(shape == (100, 200, 3) for shape in calls) == 12, "全图检测仍须逐帧执行"
    assert any(shape != (100, 200, 3) for shape in calls), "延长遮挡须进行实际局部检测"


def test_track_state_is_bounded_without_discarding_current_masks():
    from vision_shield.redaction import FrameRedactor

    boxes = [(2 * (i % 20), 2 * (i // 20), 2 * (i % 20) + 1, 2 * (i // 20) + 1) for i in range(150)]
    class ManyTargets:
        def detect(self, pixels):
            return boxes

    redactor = FrameRedactor(ManyTargets())
    actual = redactor.apply(np.full((40, 40, 3), 180, np.uint8), 0)
    assert len(redactor.history) == 128
    assert all((actual[y1:y2, x1:x2] == 64).all() for x1, y1, x2, y2 in boxes)


def test_failed_motion_match_rechecks_current_position(monkeypatch):
    from vision_shield.redaction import FrameRedactor

    class Once:
        calls = 0

        def detect(self, pixels):
            self.calls += 1
            return [(20, 20, 40, 40)] if self.calls == 1 else []

    redactor = FrameRedactor(Once())
    pixels = np.full((100, 100, 3), 180, np.uint8)
    redactor.apply(pixels, 0)
    monkeypatch.setattr(redactor, "_follow", lambda gray, box: None)
    monkeypatch.setattr(redactor, "_recheck", lambda pixels, box: (30, 20, 50, 40))
    actual = redactor.apply(pixels, 0.05)
    assert (actual[20:40, 30:50] == 64).all()
    assert actual[25, 22].min() == 180
    assert redactor.history[0][0] == 0.05
