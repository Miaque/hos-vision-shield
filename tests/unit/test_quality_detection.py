import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.detection import Detection


def test_quality_is_default_and_requires_both_primary_and_detail_models(monkeypatch):
    from vision_shield.detection import load_detector, quality

    received = []
    primary = object()
    detail = object()
    monkeypatch.setattr(quality, "PrimaryDetector", lambda *args, **kwargs: received.append(("primary", args, kwargs)) or primary)
    monkeypatch.setattr(quality, "DetailDetector", lambda *args, **kwargs: received.append(("detail", args, kwargs)) or detail)
    settings = Settings(_env_file=None)
    assert settings.detection_backend == "quality"
    detector = load_detector(settings.model_dump())
    assert detector.primary is primary and detector.detail is detail
    assert received[0][1][:2] == (settings.egoblur_face_model_path, settings.egoblur_plate_model_path)
    assert received[0][2] == {"face_threshold": 0.1}
    assert received[1][2] == {"face_threshold": 0.3}


def test_quality_keeps_strong_primary_and_confirmed_wider_edges():
    from vision_shield.detection.quality import Detector

    class Fixed:
        def __init__(self, boxes):
            self.boxes = boxes

        def candidates(self, rgb, *, kind=None):
            return self.boxes

    detector = Detector.__new__(Detector)
    detector.box_padding = 0
    detector.primary = Fixed([Detection("face", (2, 2, 8, 6), 0.8), Detection("plate", (30, 3, 39, 8), 0.8)])
    detector.detail = Fixed([Detection("face", (0, 1, 10, 7), 0.7)])
    rgb = np.full((10, 40, 3), 200, np.uint8)
    from vision_shield.redaction import redact
    masked = redact(rgb, detector.detect(rgb))
    assert not masked[1:7, :10].any()
    assert (masked[3:8, 15:18] == 200).all()
    assert not masked[3:8, 30:39].any()
    assert (rgb == 200).all()


@pytest.mark.parametrize("failed", ["primary", "detail"])
def test_any_quality_model_failure_prevents_partial_success(failed):
    from vision_shield.detection.quality import Detector

    class Model:
        def __init__(self, fails):
            self.fails = fails

        def candidates(self, rgb, *, kind=None):
            if self.fails:
                raise RuntimeError("model failed")
            return [Detection("face", (1, 1, 3, 3), 0.9)]

    detector = Detector.__new__(Detector)
    detector.primary, detector.detail = Model(failed == "primary"), Model(failed == "detail")
    with pytest.raises(RuntimeError, match="model failed"):
        detector.detect(np.zeros((8, 8, 3), np.uint8))


def test_unconfirmed_wheel_face_and_background_plate_are_rejected():
    from vision_shield.detection.quality import Detector

    rgb = np.full((200, 300, 3), 180, np.uint8)

    class Primary:
        def candidates(self, pixels, *, kind=None):
            return [Detection("plate", (20, 40, 80, 70), 0.5)] if pixels is rgb else []

    class Detail:
        def candidates(self, pixels, *, kind=None):
            return [Detection("face", (150, 120, 210, 190), 0.76)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.box_padding, detector.primary, detector.detail = 0.15, Primary(), Detail()
    assert detector.detect(rgb) == []
    assert (rgb == 180).all()


@pytest.mark.parametrize("score,local_match,expected", [(0.23, False, False), (0.23, True, True), (0.31, False, True)])
def test_weak_global_face_support_requires_local_confirmation(score, local_match, expected):
    from vision_shield.detection.quality import Detector

    rgb = np.full((10, 40, 3), 180, np.uint8)
    local_calls = []

    class Primary:
        def candidates(self, pixels, *, kind=None):
            if pixels is rgb:
                return [Detection("face", (1, 2, 9, 5), score)]
            local_calls.append(pixels.shape)
            return [Detection("face", (8, 16, 72, 40), 0.2)] if local_match else []

    class Detail:
        def candidates(self, pixels, *, kind=None):
            return [Detection("face", (1, 2, 9, 5), 0.63)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.box_padding, detector.primary, detector.detail = 0, Primary(), Detail()
    assert bool(detector.detect(rgb)) is expected
    assert len(local_calls) == (1 if score < 0.3 else 0)


@pytest.mark.parametrize("primary_score,detail_score,short_side,local_primary_score,expected", [
    (0.47, 0.76, 128, 0.38, False), (0.665, 0.801, 128, 0.38, False),
    (0.665, 0.801, 127, 0.38, True), (0.23, 0.95, 128, 0.2, True),
])
def test_large_uncertain_faces_require_stronger_support(primary_score, detail_score, short_side, local_primary_score, expected):
    from vision_shield.detection.quality import Detector

    rgb = np.full((200, 300, 3), 180, np.uint8)

    class Model:
        def __init__(self, score, local_score):
            self.score, self.local_score = score, local_score

        def candidates(self, pixels, *, kind=None):
            box = (50, 30, 50 + short_side, 190) if pixels is rgb else (
                50 * 320 / 300, 30 * 213 / 200, (50 + short_side) * 320 / 300, 190 * 213 / 200)
            return [Detection("face", box, self.score if pixels is rgb else self.local_score)]

    detector = Detector.__new__(Detector)
    detector.box_padding = 0
    detector.primary, detector.detail = Model(primary_score, local_primary_score), Model(detail_score, 0.75)
    assert bool(detector.detect(rgb)) is expected


@pytest.mark.parametrize("confirmation", ["plate", "face", "elsewhere", "failure"])
def test_local_confirmation_masks_only_matching_target_and_propagates_failure(confirmation):
    from vision_shield.detection.quality import Detector

    rgb = np.full((10, 40, 3), 180, np.uint8)
    rgb.setflags(write=False)

    class Primary:
        def candidates(self, pixels, *, kind=None):
            return [Detection("plate", (1, 2, 9, 5), 0.37)] if pixels is rgb else []

    class Detail:
        def candidates(self, pixels, *, kind=None):
            if pixels is rgb:
                return []
            assert pixels.shape == (80, 320, 3)
            if confirmation == "failure":
                raise RuntimeError("verification failed")
            box = (240, 16, 304, 40) if confirmation == "elsewhere" else (8, 16, 72, 40)
            return [Detection("face" if confirmation == "face" else "plate", box, 0.7)]

    detector = Detector.__new__(Detector)
    detector.box_padding, detector.primary, detector.detail = 0, Primary(), Detail()
    if confirmation == "failure":
        with pytest.raises(RuntimeError, match="verification failed"):
            detector.detect(rgb)
    else:
        assert detector.detect(rgb) == ([(1, 2, 9, 5)] if confirmation == "plate" else [])
    assert (rgb == 180).all()


def test_primary_face_filter_does_not_discard_low_confidence_plates():
    import torch
    from vision_shield.detection.egoblur import Detector

    def face(rgb):
        return torch.tensor([[5., 5., 30., 30.], [40., 5., 50., 20.]]), None, torch.tensor([0.14, 0.8]), None

    def plate(rgb):
        return torch.tensor([[5., 40., 30., 50.]]), None, torch.tensor([0.14]), None

    detector = Detector.__new__(Detector)
    detector.models, detector.face_threshold = [face, plate], 0.3
    detector.max_side, detector.box_padding = 1280, 0
    assert detector.detect(np.zeros((64, 64, 3), np.uint8)) == [(40., 5., 50., 20.), (5., 40., 30., 50.)]


@pytest.mark.parametrize("kind", ["face", "plate"])
def test_primary_can_run_only_the_requested_confirmation_model(kind):
    import torch
    from vision_shield.detection.egoblur import Detector

    def selected(rgb):
        return torch.tensor([[1., 2., 9., 5.]]), None, torch.tensor([0.8]), None

    def unrelated(rgb):
        raise AssertionError("局部复核不应运行无关类别模型")

    detector = Detector.__new__(Detector)
    detector.models = [selected, unrelated] if kind == "face" else [unrelated, selected]
    detector.face_threshold, detector.max_side, detector.box_padding = 0.1, 1280, 0.15
    result = detector.candidates(np.zeros((10, 40, 3), np.uint8), kind=kind)
    assert len(result) == 1 and result[0].kind == kind
    assert result[0].box == (1., 2., 9., 5.)


@pytest.mark.parametrize("kind", ["face", "plate"])
def test_local_confirmation_requests_only_the_candidate_class(kind):
    from vision_shield.detection.quality import Detector

    class Verifier:
        def candidates(self, pixels, *, kind=None):
            assert kind == expected_kind, "必须向底层模型传递复核类别"
            return [Detection(kind, (8, 16, 72, 40), 0.8)]

    expected_kind = kind
    detector = Detector.__new__(Detector)
    assert detector._confirmed(np.zeros((10, 40, 3), np.uint8),
                               Detection(kind, (1, 2, 9, 5), 0.5), Verifier())


def test_confirmation_at_bottom_right_keeps_source_offsets():
    from vision_shield.detection.quality import Detector

    rgb = np.zeros((55, 101, 3), np.uint8)
    rgb[:, :, 0] = np.arange(55)[:, None]
    rgb[:, :, 1] = np.arange(101)[None, :]

    class Primary:
        def candidates(self, pixels, *, kind=None):
            return [Detection("plate", (95, 49, 101, 55), 0.37)] if pixels is rgb else []

    class Detail:
        def candidates(self, pixels, *, kind=None):
            if pixels is rgb:
                return []
            assert pixels.shape == (320, 320, 3)
            assert pixels[0, 0, :2].tolist() == [4, 50]
            return [Detection("plate", (45 * 320 / 51, 45 * 320 / 51, 320, 320), 0.8)]

    detector = Detector.__new__(Detector)
    detector.box_padding, detector.primary, detector.detail = 0, Primary(), Detail()
    assert detector.detect(rgb) == [(95, 49, 101, 55)]


@pytest.mark.parametrize("bad", ["dimensions", "coordinates", "scores"])
def test_invalid_primary_metadata_cannot_be_used_as_confirmation(bad):
    import torch
    from vision_shield.detection.egoblur import Detector

    def model(rgb):
        box = [2., 2., 1., 3.] if bad == "coordinates" else [1., 1., 3., 3.]
        boxes = torch.tensor([box])
        return boxes[:, :3] if bad == "dimensions" else boxes, None, torch.tensor([1.5 if bad == "scores" else 0.9]), None

    detector = Detector.__new__(Detector)
    detector.models, detector.face_threshold = [model], 0.1
    detector.max_side, detector.box_padding = 1280, 0.15
    with pytest.raises(ValueError, match="Invalid detector output"):
        detector.candidates(np.zeros((10, 10, 3), np.uint8))
