import numpy as np
import pytest

from vision_shield.detection import Detection
from vision_shield.redaction import FrameRedactor


class Detections:
    boxes = []

    def detect(self, pixels):
        return self.boxes


def mask_bounds(pixels):
    rows, columns = np.nonzero(np.all(pixels == 64, axis=2))
    assert len(rows), "本帧应包含遮挡"
    return columns.min(), rows.min(), columns.max() + 1, rows.max() + 1


def test_mask_shrinks_gradually_without_losing_current_detection():
    detector = Detections()
    redactor = FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    detector.boxes = [(30, 30, 70, 70)]
    redactor.apply(pixels, 0)
    detector.boxes = [(34, 34, 66, 66)]
    output = redactor.apply(pixels, 0.04)
    left, top, right, bottom = mask_bounds(output)
    assert 32 < right - left <= 40, "连续检测框收缩不能立即把遮挡边缘跳回去"
    assert 32 < bottom - top <= 40
    for index in range(2, 31):
        output = redactor.apply(pixels, index / 25)
        assert (output[34:66, 34:66] == 64).all(), "平滑不得缩进当帧可信检测框"
    left, top, right, bottom = mask_bounds(output)
    assert right - left <= 34 and bottom - top <= 34, "稳定小目标不应永远保留旧大框"
    assert np.all(pixels == 180), "不能改写原始帧供下一帧跟踪"


def test_stationary_target_has_less_edge_jitter_than_alternating_detections():
    detector = Detections()
    redactor = FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    edges = []
    for index in range(30):
        left = 30 if index % 2 else 34
        detector.boxes = [(left, 30, left + 30, 60)]
        output = redactor.apply(pixels, index / 25)
        assert (output[30:60, left:left + 30] == 64).all()
        edges.append(mask_bounds(output))
    steady = np.asarray(edges[10:])
    assert np.ptp(steady[:, 0]) < 4, "静止目标左边缘不应完整复现检测器的4像素跳变"
    assert np.ptp(steady[:, 2]) < 4, "静止目标右边缘不应完整复现检测器的4像素跳变"


def test_smoothing_never_exposes_fast_moving_detection_or_leaves_unbounded_trail():
    detector = Detections()
    redactor = FrameRedactor(detector)
    texture = np.random.default_rng(23).integers(40, 220, (20, 20, 3), dtype=np.uint8)
    for index in range(20):
        left = 20 + index * 7
        pixels = np.full((100, 200, 3), 180, np.uint8)
        pixels[30:50, left:left + 20] = texture
        detector.boxes = [(left, 30, left + 20, 50)]
        original = pixels.copy()
        output = redactor.apply(pixels, index / 25)
        assert (output[30:50, left:left + 20] == 64).all(), "平滑不能落在运动目标后面而露出前缘"
        assert mask_bounds(output)[0] >= left - 20, "运动遮挡不能累积成沿整个路径的黑色拖尾"
        assert np.array_equal(pixels, original)


def test_smoothing_does_not_extend_a_disappeared_target_indefinitely():
    detector = Detections()
    redactor = FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    for index in range(10):
        edge = 30 if index % 2 else 34
        detector.boxes = [(edge, edge, 100 - edge, 100 - edge)]
        redactor.apply(pixels, index / 25)
    detector.boxes = []
    for index in range(10, 31):
        output = redactor.apply(pixels, index / 25)
    assert np.array_equal(output, pixels), "无新检测时，平滑状态不能无限续命"


def test_plate_cannot_renew_an_overlapping_face_during_local_recheck():
    class TypedDetections:
        first = True

        def detect_regions(self, pixels, *, kind=None):
            if self.first:
                self.first = False
                return [Detection("face", (20, 20, 60, 60), 0.9)]
            if pixels.shape == (100, 100, 3):
                return [Detection("plate", (30, 30, 50, 50), 0.9)]
            # The 100px source crop is enlarged to 320px for local verification.
            return [Detection("plate", (64, 64, 192, 192), 0.9)]

    redactor = FrameRedactor(TypedDetections())
    pixels = np.full((100, 100, 3), 180, np.uint8)
    for index in range(17):
        output = redactor.apply(pixels, index / 25)
    assert (output[30:50, 30:50] == 64).all()
    assert np.array_equal(output[20:30, 20:30], pixels[20:30, 20:30]), "车牌局部命中不能为旧人脸续期"


def test_one_detection_does_not_discard_a_second_overlapping_target():
    detector = Detections()
    redactor = FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    pixels[30:60, 20:70] = np.random.default_rng(51).integers(40, 220, (30, 50, 3), dtype=np.uint8)
    detector.boxes = [(20, 30, 50, 60), (40, 30, 70, 60)]
    redactor.apply(pixels, 0)
    detector.boxes = [(22, 30, 52, 60)]
    output = redactor.apply(pixels, 0.04)
    assert (output[30:60, 22:52] == 64).all()
    assert (output[31:59, 53:69] == 64).all(), "已关联到左侧目标的检测不能再取消右侧目标的短时遮挡"


@pytest.mark.parametrize("detection", [
    Detection("face", (20, 20, float("nan"), 40), 0.9),
    Detection("face", (20, 20, 40, 40), 1.1),
    Detection("person", (20, 20, 40, 40), 0.9),
])
def test_invalid_typed_output_is_not_published_as_a_successful_frame(detection):
    class Invalid:
        def detect_regions(self, pixels, *, kind=None):
            return [detection]

    with pytest.raises(ValueError, match="Invalid detector output"):
        FrameRedactor(Invalid()).apply(np.full((100, 100, 3), 180, np.uint8), 0)


def test_local_recheck_error_propagates_instead_of_returning_an_unmasked_frame():
    class FailedRecheck:
        first = True

        def detect_regions(self, pixels, *, kind=None):
            if pixels.shape != (100, 100, 3):
                raise RuntimeError("local inference failed")
            if self.first:
                self.first = False
                return [Detection("face", (20, 20, 40, 40), 0.9)]
            return []

    pixels = np.full((100, 100, 3), 180, np.uint8)
    redactor = FrameRedactor(FailedRecheck())
    redactor.apply(pixels, 0)
    with pytest.raises(RuntimeError, match="local inference failed"):
        redactor.apply(pixels, 0.04)


def test_fast_small_face_does_not_accumulate_masks_at_previous_positions():
    detector = Detections()
    redactor = FrameRedactor(detector)
    texture = np.random.default_rng(52).integers(40, 220, (30, 24, 3), dtype=np.uint8)
    for index in range(8):
        left = 30 + index * 40
        pixels = np.full((100, 400, 3), 180, np.uint8)
        pixels[30:60, left:left + 24] = texture
        detector.boxes = [(left, 30, left + 24, 60)]
        output = redactor.apply(pixels, index / 20)
        assert (output[30:60, left:left + 24] == 64).all()
        assert np.array_equal(output[:, :left - 6], pixels[:, :left - 6])
        assert len(redactor.history) == 1


def test_overlapping_model_detections_share_one_track_and_preserve_all_edges():
    detector = Detections()
    redactor = FrameRedactor(detector)
    pixels = np.full((100, 100, 3), 180, np.uint8)
    for index in range(6):
        detector.boxes = [(20, 20, 40, 40), (22, 21, 42, 41)] if index % 2 == 0 else [(21, 20, 41, 40)]
        output = redactor.apply(pixels, index / 20)
        for x, y, right, bottom in detector.boxes:
            assert (output[y:bottom, x:right] == 64).all()
        assert len(redactor.history) == 1


def test_identical_face_and_plate_boxes_remain_separate_tracks():
    detections = [Detection(kind, (20, 20, 40, 40), 0.9) for kind in ("face", "plate")]
    assert FrameRedactor._merge_detections(detections) == detections


def test_repeated_texture_cannot_move_a_track_to_a_different_copy():
    redactor = FrameRedactor(Detections())
    previous = np.full((100, 300), 180, np.uint8)
    current = previous.copy()
    texture = np.random.default_rng(17).integers(40, 220, (20, 20), dtype=np.uint8)
    for left in (40, 120):
        previous[30:50, left:left + 20] = texture
    for left in (80, 160):
        current[30:50, left:left + 20] = texture
    redactor.gray = previous
    assert redactor._follow(current, (120, 30, 140, 50)) is None


def test_lost_face_cannot_start_following_the_background_in_later_frames():
    rng = np.random.default_rng(71)
    background = rng.integers(150, 210, (100, 200, 3), dtype=np.uint8)
    first = background.copy()
    first[30:60, 20:50] = rng.integers(30, 130, (30, 30, 3), dtype=np.uint8)
    detector = Detections()
    redactor = FrameRedactor(detector)
    detector.boxes = [(20, 30, 50, 60)]
    redactor.apply(first, 0)
    detector.boxes = []
    held = redactor.apply(background, 0.05)
    assert (held[30:60, 20:50] == 64).all(), "首次跟丢仍保留原短时保护"
    for index in range(2, 7):
        output = redactor.apply(background, index / 20)
    assert np.array_equal(output, background), "跟丢后旧位置变为背景，不能将背景匹配当作连续跟踪"
    assert not redactor.history


def test_new_detection_restores_following_after_a_lost_frame():
    rng = np.random.default_rng(72)
    background = rng.integers(150, 210, (100, 200, 3), dtype=np.uint8)
    texture = rng.integers(30, 130, (30, 30, 3), dtype=np.uint8)
    detector = Detections()
    redactor = FrameRedactor(detector)
    first = background.copy()
    first[30:60, 20:50] = texture
    detector.boxes = [(20, 30, 50, 60)]
    redactor.apply(first, 0)
    detector.boxes = []
    redactor.apply(background, 0.05)
    detector.boxes = [(20, 30, 50, 60)]
    redactor.apply(first, 0.1)
    moved = background.copy()
    moved[30:60, 60:90] = texture
    detector.boxes = []
    output = redactor.apply(moved, 0.15)
    assert (output[30:60, 60:90] == 64).all(), "实际检测重新命中后，应继续跟随移动目标而非永久禁用跟踪"


def test_briefly_occluded_face_can_resume_visual_tracking_without_a_new_detection():
    rng = np.random.default_rng(73)
    background = rng.integers(150, 210, (100, 200, 3), dtype=np.uint8)
    texture = rng.integers(30, 130, (30, 30, 3), dtype=np.uint8)
    first = background.copy()
    first[30:60, 20:50] = texture
    detector = Detections()
    redactor = FrameRedactor(detector)
    detector.boxes = [(20, 30, 50, 60)]
    redactor.apply(first, 0)
    detector.boxes = []
    redactor.apply(background, 0.05)
    redactor.apply(first, 0.1)
    moved = background.copy()
    moved[30:60, 60:90] = texture
    for moment in (0.15, 0.2, 0.25, 0.3):
        output = redactor.apply(moved, moment)
        assert (output[30:60, 60:90] == 64).all(), "短暂遮挡后仍应通过原目标纹理重获移动人脸"
        assert np.array_equal(output[:, :54], moved[:, :54]), "恢复移动跟踪后旧位置不能留下余框"
