from types import MethodType

import numpy as np
import pytest


def detector_with(predict, *, max_side=1280, padding=0):
    from vision_shield.detection.cpu import Detector

    detector = Detector.__new__(Detector)
    detector.max_side, detector.box_padding = max_side, padding
    detector.tile_side = 640
    detector._detect_view = MethodType(lambda self, view, **kwargs: predict(view), detector)
    return detector


def test_rgb_input_can_be_read_only_and_model_sees_bgr_without_changing_source():
    pixels = np.full((80, 120, 3), [11, 23, 47], dtype=np.uint8)
    original = pixels.copy()
    pixels.setflags(write=False)
    seen = []

    def predict(view):
        seen.append(view)
        assert view.dtype == np.uint8
        assert np.array_equal(view[0, 0], [47, 23, 11])
        view[0, 0] = 0
        return []

    assert detector_with(predict).detect(pixels) == []
    assert seen
    assert np.array_equal(pixels, original)


def test_whole_view_maps_each_axis_using_actual_rounded_dimensions():
    from vision_shield.detection.cpu import Detection

    pixels = np.zeros((541, 2001, 3), dtype=np.uint8)
    seen = []

    def predict(view):
        seen.append(view.shape)
        if len(seen) == 1:
            return [Detection("face", (100, 20, 200, 80), 0.9)]
        return []

    boxes = detector_with(predict).detect(pixels)
    assert seen[0] == (346, 1280, 3)
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx((100 * 2001 / 1280, 20 * 541 / 346,
                                      200 * 2001 / 1280, 80 * 541 / 346))


def test_source_slices_find_small_target_at_right_bottom_edge_with_bounded_work():
    from vision_shield.detection.cpu import Detection

    pixels = np.zeros((1081, 3841, 3), dtype=np.uint8)
    pixels[-8:, -10:] = [11, 23, 47]
    shapes = []

    def predict(view):
        shapes.append(view.shape)
        if len(shapes) == 1:
            return []
        rows, columns = np.where(np.all(view == [47, 23, 11], axis=2))
        if not len(rows):
            return []
        return [Detection("plate", (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1), 0.9)]

    boxes = detector_with(predict).detect(pixels)
    assert any(np.allclose(box, (3831, 1073, 3841, 1081)) for box in boxes)
    assert len(shapes) <= 64
    assert all(max(shape[:2]) <= 1280 for shape in shapes)


def test_empty_predictions_produce_no_masks():
    pixels = np.zeros((1080, 3840, 3), dtype=np.uint8)
    assert detector_with(lambda view: []).detect(pixels) == []


def test_global_only_detail_uses_one_view_and_keeps_full_resolution_box_mapping():
    from vision_shield.detection import Detection

    seen = []

    def predict(view):
        seen.append(view.shape)
        return [Detection("face", (100, 20, 200, 80), 0.9)]

    detector = detector_with(predict)
    detector.tile_side = 1920
    boxes = detector.detect(np.zeros((536, 1920, 3), np.uint8))
    assert seen == [(357, 1280, 3)]
    assert boxes == [pytest.approx((150, 20 * 536 / 357, 300, 80 * 536 / 357))]


def test_large_detail_tiles_cover_ultrawide_image_and_keep_bottom_right_target():
    from vision_shield.detection import Detection

    pixels = np.zeros((1080, 3840, 3), np.uint8)
    pixels[-30:, -30:] = [11, 23, 47]
    shapes = []

    def predict(view):
        shapes.append(view.shape)
        rows, columns = np.where(np.all(view == [47, 23, 11], axis=2))
        if not len(rows):
            return []
        return [Detection("plate", (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1), 0.9)]

    detector = detector_with(predict)
    detector.tile_side = 1920
    boxes = detector.detect(pixels)
    assert len(shapes) == 4
    assert any(np.allclose(box, (3810, 1050, 3840, 1080), atol=2) for box in boxes)
    coverage = np.zeros((1080, 3840), dtype=bool)
    for left, top, right, bottom in detector._views(3840, 1080, 1920):
        if (left, top, right, bottom) != (0, 0, 3840, 1080):
            coverage[top:bottom, left:right] = True
    assert coverage.all()


@pytest.mark.parametrize("box,score", [
    ((float("nan"), 10, 30, 40), 0.9),
    ((10, 10, float("inf"), 40), 0.9),
    ((30, 10, 10, 40), 0.9),
    ((10, 40, 30, 10), 0.9),
    ((10, 10, 30, 40), float("nan")),
    ((10, 10, 30, 40), float("inf")),
])
def test_invalid_model_output_cannot_succeed_with_a_missing_mask(box, score):
    from vision_shield.detection.cpu import Detection

    detector = detector_with(lambda view: [Detection("plate", box, score)])
    with pytest.raises(ValueError):
        detector.detect(np.zeros((80, 120, 3), dtype=np.uint8))


def test_face_and_plate_margins_differ_and_are_clipped_to_the_source():
    from vision_shield.detection.cpu import Detection

    detector = detector_with(lambda view: [Detection("face", (20, 5, 60, 45), 0.9),
                                           Detection("plate", (20, 5, 60, 45), 0.9)], padding=0.15)
    assert detector.detect(np.zeros((50, 70, 3), dtype=np.uint8)) == [
        (14, 0, 66, 50), (8, 0, 70, 50),
    ]
    assert detector.candidates(np.zeros((50, 70, 3), dtype=np.uint8)) == [
        Detection("face", (20, 5, 60, 45), 0.9), Detection("plate", (20, 5, 60, 45), 0.9),
    ]


@pytest.mark.parametrize("width,height", [(10**9, 1), (1, 10**9), (10**9, 10**9)])
@pytest.mark.parametrize("tile_side", [640, 1920])
def test_extreme_source_views_cover_every_region_with_a_bounded_grid(width, height, tile_side):
    from vision_shield.detection.cpu import Detector

    views = list(Detector._views(width, height, tile_side))
    assert views[0] == (0, 0, width, height)
    tiles = views[1:]
    assert 1 < len(views) <= 65
    assert len({left for left, _, _, _ in tiles}) <= 8
    assert len({top for _, top, _, _ in tiles}) <= 8
    assert all(0 <= left < right <= width and 0 <= top < bottom <= height
               for left, top, right, bottom in tiles)
    x_edges = sorted({0, width, *(x for left, _, right, _ in tiles for x in (left, right))})
    y_edges = sorted({0, height, *(y for _, top, _, bottom in tiles for y in (top, bottom))})
    for x1, x2 in zip(x_edges, x_edges[1:]):
        for y1, y2 in zip(y_edges, y_edges[1:]):
            assert any(left <= x1 < x2 <= right and top <= y1 < y2 <= bottom
                       for left, top, right, bottom in tiles)


def detector_with_raw_outputs(faces, plates):
    from vision_shield.detection.cpu import Detector

    class Face:
        def setInputSize(self, size):
            pass

        def detect(self, view):
            return None, faces

    class Plate:
        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            return plates

    detector = Detector.__new__(Detector)
    detector.tile_side = 640
    detector.face, detector.plate = Face(), Plate()
    detector.parallel = False
    return detector


def test_raw_detectors_map_letterbox_with_actual_axis_scales_and_no_ocr():
    faces = np.zeros((1, 15), dtype=np.float32)
    faces[0, :4], faces[0, -1] = [10, 20, 30, 40], 0.9
    plates = np.zeros((1, 6300, 15), dtype=np.float32)
    plates[0, 0, :5] = [160, 160, 64, 20, 0.8]
    plates[0, 0, 13:15] = [0.25, 0.9]
    detector = detector_with_raw_outputs(faces, plates)
    detections = detector._detect_view(np.full((333, 997, 3), [17, 31, 61], np.uint8))

    face, plate = sorted(detections, key=lambda detection: detection.kind)
    assert face.kind == "face" and face.box == (10, 20, 40, 60)
    assert plate.kind == "plate"
    assert plate.box == pytest.approx((128 * 997 / 320, (150 - 106) * 333 / 107,
                                       192 * 997 / 320, (170 - 106) * 333 / 107))
    assert plate.score == pytest.approx(0.8 * 0.9)
    assert detector.plate.blob.shape == (1, 3, 320, 320)
    assert detector.plate.blob[0, :, 160, 160] == pytest.approx(np.array([61, 31, 17]) / 255)
    assert not detector.plate.blob[0, :, 0, 0].any()


@pytest.mark.parametrize("kind", ["face", "plate"])
def test_local_cpu_confirmation_avoids_the_unrelated_model(kind):
    faces = np.zeros((1, 15), dtype=np.float32)
    faces[0, :4], faces[0, -1] = [10, 20, 30, 40], 0.9
    plates = np.zeros((1, 6300, 15), dtype=np.float32)
    plates[0, 0, :5], plates[0, 0, 13] = [160, 160, 64, 20, 0.9], 0.9
    detector = detector_with_raw_outputs(faces, plates)
    pixels = np.zeros((333, 497, 3), dtype=np.uint8)
    expected = [d for d in detector._detect_view(pixels) if d.kind == kind]

    class Unrelated:
        def __getattr__(self, name):
            raise AssertionError("局部复核不应运行无关类别模型")

    if kind == "face":
        detector.plate = Unrelated()
    else:
        detector.face = Unrelated()
    detector.max_side, detector.box_padding = 1280, 0.15
    assert detector.candidates(pixels, kind=kind) == expected


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("failure", ["face_shape", "face_nan", "plate_shape", "plate_nan", "plate_dimensions"])
def test_invalid_raw_model_outputs_fail_before_producing_masks(failure, parallel):
    faces = np.zeros((1, 15), dtype=np.float32)
    plates = np.zeros((1, 6300, 15), dtype=np.float32)
    if failure == "face_shape":
        faces = faces[:, :14]
    elif failure == "face_nan":
        faces[0, 0] = np.nan
    elif failure == "plate_shape":
        plates = plates[:, :, :14]
    elif failure == "plate_nan":
        plates[0, 0, 0] = np.nan
    else:
        plates[0, 0, :5] = [160, 160, -20, 20, 0.9]
        plates[0, 0, 13] = 0.9
    detector = detector_with_raw_outputs(faces, plates)
    detector.parallel = parallel
    with pytest.raises(ValueError):
        detector._detect_view(np.zeros((333, 997, 3), dtype=np.uint8))


@pytest.mark.parametrize("failed", [None, "face", "plate"])
def test_parallel_view_preserves_order_and_joins_both_models_on_failure(failed):
    from threading import Event

    faces = np.zeros((1, 15), dtype=np.float32)
    faces[0, :4], faces[0, -1] = [10, 20, 30, 40], 0.9
    plates = np.zeros((1, 6300, 15), dtype=np.float32)
    plates[0, 0, :5], plates[0, 0, 13] = [160, 160, 64, 20, 0.9], 0.9
    detector = detector_with_raw_outputs(faces, plates)
    pixels = np.full((64, 64, 3), 180, dtype=np.uint8)
    pixels.setflags(write=False)
    expected = detector._detect_view(pixels, kind="face") + detector._detect_view(pixels, kind="plate")
    face_started, plate_started, face_finished = Event(), Event(), Event()

    class Face(type(detector.face)):
        def detect(self, view):
            face_started.set()
            try:
                assert plate_started.wait(2), "独立的人脸和车牌模型应重叠执行"
                if failed == "face":
                    raise RuntimeError("face failed")
                return super().detect(view)
            finally:
                face_finished.set()

    class Plate(type(detector.plate)):
        def forward(self):
            assert face_started.wait(2)
            plate_started.set()
            if failed == "plate":
                raise RuntimeError("plate failed")
            return super().forward()

    detector.face, detector.plate, detector.parallel = Face(), Plate(), True
    if failed:
        with pytest.raises(RuntimeError, match=f"{failed} failed"):
            detector._detect_view(pixels)
    else:
        assert detector._detect_view(pixels) == expected
    assert face_finished.is_set(), "失败或正常结束前都须等待后台模型结束"
    assert (pixels == 180).all()
