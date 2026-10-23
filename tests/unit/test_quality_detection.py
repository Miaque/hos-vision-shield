import numpy as np
import pytest

from vision_shield.config import Settings
from vision_shield.detection import Detection


def test_candidate_timings_keep_each_model_span_without_changing_masks(monkeypatch):
    from vision_shield.detection import quality

    clock = [0.0]
    monkeypatch.setattr(quality.time, "monotonic", lambda: clock[0])

    class Fixed:
        def __init__(self, seconds, regions):
            self.seconds, self.regions = seconds, regions

        def candidates(self, pixels, *, kind=None):
            clock[0] += self.seconds
            return self.regions

    detector = quality.Detector.__new__(quality.Detector)
    detector.inference_device, detector.box_padding = "cpu", 0
    region = Detection("face", (2, 2, 8, 8), 0.9)
    detector.primary, detector.detail = Fixed(0.03, [region]), Fixed(0.02, [])
    pixels = np.full((12, 12, 3), 180, np.uint8)
    assert detector.detect_regions(pixels) == [region]
    assert detector.candidate_timings == pytest.approx({"candidate-primary": 0.03, "candidate-detail": 0.02})
    assert (pixels == 180).all()


def test_cuda_candidates_share_one_prepared_frame():
    from types import SimpleNamespace
    from vision_shield.detection.quality import Detector

    source = np.full((12, 12, 3), 180, np.uint8)
    prepared = source.copy()
    preparations, received = [], []

    def prepare(pixels):
        assert pixels is source
        preparations.append(pixels)
        return prepared

    def candidates(pixels, *, kind=None):
        received.append(pixels)
        assert pixels is prepared
        return []

    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = "cuda", 0
    detector.primary = SimpleNamespace(prepare=prepare, candidates=candidates)
    detector.detail = SimpleNamespace(candidates=candidates)
    assert detector.detect_regions(source) == []
    assert len(preparations) == 1 and len(received) == 2


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
    assert received[0][2] == {"face_threshold": 0.1, "device": "cpu"}
    assert received[1][2] == {"face_threshold": 0.3, "parallel": False, "tile_side": 1920}


@pytest.mark.parametrize("failed", [None, "primary", "detail"])
def test_cuda_overlaps_independent_models_and_waits_for_failure(monkeypatch, failed):
    from threading import Event

    from vision_shield.detection import quality

    primary_started, detail_started, primary_finished = Event(), Event(), Event()

    class Primary:
        def candidates(self, rgb, *, kind=None):
            primary_started.set()
            try:
                assert detail_started.wait(2), "两路全图检测应重叠执行"
                if failed == "primary":
                    raise RuntimeError("主检测失败")
                return [Detection("face", (1, 1, 3, 3), 0.9)]
            finally:
                primary_finished.set()

    class Detail:
        def candidates(self, rgb, *, kind=None):
            assert primary_started.wait(2)
            detail_started.set()
            if failed == "detail":
                raise RuntimeError("补充检测失败")
            return []

    monkeypatch.setattr(quality, "PrimaryDetector", lambda *a, **kw: Primary())

    def detail_detector(*args, **kwargs):
        assert kwargs["parallel"] is True
        assert kwargs["tile_side"] == 1920
        return Detail()

    from vision_shield.detection import cuda_detail
    monkeypatch.setattr(cuda_detail, "Detector", detail_detector)
    detector = quality.Detector(Settings(_env_file=None, inference_device="cuda", detection_box_padding=0).model_dump())
    pixels = np.full((8, 8, 3), 180, np.uint8)
    pixels.setflags(write=False)
    if failed:
        with pytest.raises(RuntimeError, match="检测失败"):
            detector.detect(pixels)
    else:
        assert detector.detect(pixels) == [(1, 1, 3, 3)]
    assert primary_finished.is_set(), "检测结束前必须回收后台工作"
    assert (pixels == 180).all()


def test_quality_keeps_strong_primary_and_confirmed_wider_edges():
    from vision_shield.detection.quality import Detector

    class Fixed:
        def __init__(self, boxes):
            self.boxes = boxes

        def candidates(self, rgb, *, kind=None):
            return self.boxes

    detector = Detector.__new__(Detector)
    detector.inference_device = "cpu"
    detector.box_padding = 0
    detector.primary = Fixed([Detection("face", (2, 2, 8, 6), 0.8), Detection("plate", (30, 3, 39, 8), 0.8)])
    detector.detail = Fixed([Detection("face", (0, 1, 10, 7), 0.7)])
    rgb = np.full((10, 40, 3), 200, np.uint8)
    from vision_shield.redaction import redact
    masked = redact(rgb, detector.detect(rgb))
    assert (masked[1:7, :10] == 64).all()
    assert (masked[3:8, 15:18] == 200).all()
    assert (masked[3:8, 30:39] == 64).all()
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
    detector.inference_device = "cpu"
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
    detector.inference_device = "cpu"
    detector.box_padding, detector.primary, detector.detail = 0.15, Primary(), Detail()
    assert detector.detect(rgb) == []
    assert (rgb == 180).all()


def test_two_weak_global_plate_candidates_cannot_confirm_each_other():
    from vision_shield.detection.quality import Detector

    rgb = np.full((200, 300, 3), 180, np.uint8)
    class Model:
        def __init__(self, score):
            self.score = score

        def candidates(self, pixels, *, kind=None):
            return [Detection("plate", (20, 40, 80, 70), self.score)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = "cpu", 0
    detector.primary, detector.detail = Model(0.13), Model(0.34)
    assert detector.detect(rgb) == [], "两个低分全图框仍须通过局部复核"


@pytest.mark.parametrize("source,score", [("primary", 0.2), ("detail", 0.4)])
@pytest.mark.parametrize("short_side", [19.99, 20])
def test_tiny_weak_faces_do_not_trigger_enlarged_confirmation(source, score, short_side):
    from vision_shield.detection.quality import Detector

    candidate = Detection("face", (100, 100, 100 + short_side, 140), score)

    class Model:
        def __init__(self, name):
            self.name = name

        def candidates(self, pixels, *, kind=None):
            return [candidate] if self.name == source else []

    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = "cpu", 0
    detector.primary, detector.detail = Model("primary"), Model("detail")
    calls = []
    detector._recover_face = lambda *a: calls.append("recover") or True
    detector._confirmed = lambda *a, **kw: calls.append("confirm") or True
    regions = detector.detect_regions(np.zeros((320, 320, 3), np.uint8))
    assert bool(regions) is (short_side >= 20)
    assert len(calls) == (1 if short_side >= 20 else 0)


@pytest.mark.parametrize("local_score,expected", [(0.65, False), (0.8, True)])
def test_weak_small_detail_face_needs_trusted_local_primary(local_score, expected):
    from vision_shield.detection.quality import Detector

    rgb = np.full((96, 96, 3), 180, np.uint8)
    class Primary:
        def candidates(self, pixels, *, kind=None):
            return [] if pixels is rgb else [Detection("face", (0, 0, 128, 128), local_score)]
    class Detail:
        def candidates(self, pixels, *, kind=None):
            return [Detection("face", (0, 0, 24, 24), 0.4)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.primary, detector.detail = Primary(), Detail()
    detector.inference_device, detector.box_padding = "cpu", 0
    assert bool(detector.detect(rgb)) is expected


@pytest.mark.parametrize("mirror_matches", [False, True])
def test_moderate_small_face_requires_matching_normal_and_mirrored_local_views(mirror_matches):
    from vision_shield.detection.quality import Detector

    rgb = np.zeros((96, 96, 3), np.uint8)
    rgb[:, :, 0] = np.arange(96)[None, :] * 2
    class Primary:
        def candidates(self, pixels, *, kind=None):
            if pixels is rgb:
                return []
            mirrored = pixels[0, 0, 0] > 100
            box = (192, 0, 320, 128) if mirrored and mirror_matches else (0, 0, 128, 128)
            return [Detection("face", box, 0.4)]
    class Detail:
        def candidates(self, pixels, *, kind=None):
            return [Detection("face", (0, 0, 24, 24), 0.4)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.primary, detector.detail = Primary(), Detail()
    detector.inference_device, detector.box_padding = "cpu", 0
    assert bool(detector.detect(rgb)) is mirror_matches


@pytest.mark.parametrize("mode", ["matching", "wrong_class", "elsewhere", "primary_failure", "detail_failure"])
def test_weak_profile_recovery_maps_mirrored_border_view_and_propagates_errors(mode):
    from vision_shield.detection.quality import Detector

    rgb = np.zeros((200, 300, 3), np.uint8)
    rgb[:, :, 0] = np.arange(300, dtype=np.uint8)[None, :]
    rgb.setflags(write=False)
    candidate = Detection("face", (5, 30, 25, 60), 0.12)
    seen = []

    class Model:
        def __init__(self, name):
            self.name = name

        def candidates(self, pixels, *, kind=None):
            assert kind == "face"
            if mode == self.name + "_failure":
                raise RuntimeError("recovery failed")
            flipped = self.name == "detail" and pixels[0, 0, 0] > 30
            seen.append((self.name, flipped))
            if self.name == "detail" and not flipped:
                return []
            # 候选周边四倍长边在左边界截断为源坐标 (0,0)..(75,105)。
            x1, y1, x2, y2 = candidate.box
            box = (x1 * pixels.shape[1] / 75, y1 * pixels.shape[0] / 105,
                   x2 * pixels.shape[1] / 75, y2 * pixels.shape[0] / 105)
            if flipped:
                box = (pixels.shape[1] - box[2], box[1], pixels.shape[1] - box[0], box[3])
            if mode == "elsewhere" and self.name == "detail":
                box = (0, 0, 5, 5)
            return [Detection("plate" if mode == "wrong_class" else "face", box, 0.8)]

    detector = Detector.__new__(Detector)
    detector.primary, detector.detail = Model("primary"), Model("detail")
    if mode.endswith("failure"):
        with pytest.raises(RuntimeError, match="recovery failed"):
            detector._recover_face(rgb, candidate)
    else:
        assert detector._recover_face(rgb, candidate) is (mode == "matching")
        if mode == "matching":
            assert seen == [("primary", False), ("detail", False), ("detail", True)]
    assert not rgb.flags.writeable


@pytest.mark.parametrize("score,local_match,expected", [(0.23, False, False), (0.23, True, True), (0.31, False, True)])
def test_weak_global_face_support_requires_local_confirmation(score, local_match, expected):
    from vision_shield.detection.quality import Detector

    rgb = np.full((96, 96, 3), 180, np.uint8)
    local_calls = []

    class Primary:
        def candidates(self, pixels, *, kind=None):
            if pixels is rgb:
                return [Detection("face", (0, 0, 24, 24), score)]
            local_calls.append(pixels.shape)
            return [Detection("face", (0, 0, 128, 128), 0.2)] if local_match else []

    class Detail:
        def candidates(self, pixels, *, kind=None):
            return [Detection("face", (0, 0, 24, 24), 0.63)] if pixels is rgb else []

    detector = Detector.__new__(Detector)
    detector.inference_device = "cpu"
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
    detector.inference_device = "cpu"
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
    detector.inference_device = "cpu"
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
    detector.device = torch.device("cpu")
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
    detector.device = torch.device("cpu")
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
    detector.inference_device = "cpu"
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
    detector.device = torch.device("cpu")
    detector.max_side, detector.box_padding = 1280, 0.15
    with pytest.raises(ValueError, match="Invalid detector output"):
        detector.candidates(np.zeros((10, 10, 3), np.uint8))

@pytest.mark.parametrize('kind', ['face', 'plate'])
def test_typed_quality_recheck_does_not_run_unrelated_models(kind):
    from vision_shield.detection.quality import Detector

    class Model:
        def candidates(self, pixels, *, kind=None):
            assert kind == expected, '局部补检不应执行无关类别模型'
            return [Detection(kind, (20, 20, 40, 40), 0.95)]

    expected = kind
    detector = Detector.__new__(Detector)
    detector.inference_device = 'cpu'
    detector.primary = detector.detail = Model()
    detector.box_padding = 0
    regions = detector.detect_regions(np.zeros((100, 100, 3), np.uint8), kind=kind)
    assert regions and all(region.kind == kind for region in regions)
def test_flow_proposals_share_gpu_input_and_skip_expensive_primary_plate():
    from types import SimpleNamespace
    from vision_shield.detection.quality import Detector

    pixels, prepared, calls = object(), object(), []
    face = Detection("face", (20, 30, 50, 70), .4)
    plate = Detection("plate", (90, 50, 150, 80), .3)
    detector = Detector.__new__(Detector)
    detector.primary = SimpleNamespace(prepare=lambda rgb: prepared if rgb is pixels else None,
                                       proposals=lambda rgb: calls.append(("face", rgb)) or [face])
    detector.detail = SimpleNamespace(candidates=lambda rgb, *, kind: calls.append((kind, rgb)) or [plate])
    assert detector.proposals(pixels) == [face, plate]
    assert calls == [("face", prepared), ("plate", prepared)]
