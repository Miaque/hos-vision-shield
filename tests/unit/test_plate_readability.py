import cv2
import numpy as np
import pytest


def plate_frame(*, blurred=False, contrast=180):
    gray = np.full((80, 160), 30, np.uint8)
    glyphs = ["011101111001110", "100011000110001", "100011000110000", "111111111010000",
              "100011000110000", "100011000110001", "100011111001110"]
    patch = np.full((11, 39), 30, np.uint8)
    for y, row in enumerate(glyphs, 2):
        for x, value in enumerate(row, 2):
            if value == '1':
                patch[y, x] = 30 + contrast
    if blurred:
        patch = cv2.GaussianBlur(patch, (0, 0), 4)
    gray[20:31, 40:79] = patch
    return gray, (40, 20, 79, 31)


@pytest.mark.parametrize('contrast', [4, 180])
def test_small_clear_characters_are_not_skipped_even_with_low_contrast(contrast):
    from vision_shield.detection.plate_readability import needs_plate_work

    gray, box = plate_frame(contrast=contrast)
    assert needs_plate_work(gray, box)


def test_blurred_plate_can_return_to_clear_without_negative_state():
    from vision_shield.detection.plate_readability import needs_plate_work

    blurred, box = plate_frame(blurred=True)
    clear, _ = plate_frame()
    assert not needs_plate_work(blurred, box)
    assert needs_plate_work(clear, box)
    assert not needs_plate_work(blurred, box)


def test_large_smooth_plate_without_character_detail_does_not_need_recovery():
    from vision_shield.detection.plate_readability import needs_plate_work

    gray = np.full((100, 220), 50, np.uint8)
    gray[20:60, 40:160] = np.linspace(40, 70, 120).astype(np.uint8)[None, :]
    assert not needs_plate_work(gray, (40, 20, 160, 60))


@pytest.mark.parametrize('box', [(0, 20, 39, 31), (40, 20, 159, 40), (40, 0, 79, 11),
                                  (40, 69, 79, 80), (40, 20, 50, 40), (40, 20, 42, 21)])
def test_clipped_incomplete_or_insufficient_source_samples_keep_existing_work(box):
    from vision_shield.detection.plate_readability import needs_plate_work

    assert needs_plate_work(np.full((80, 160), 30, np.uint8), box)


def test_enlarged_confirmation_uses_original_pixels_and_coordinates():
    from vision_shield.detection.plate_readability import needs_plate_work, source_view

    original, box = plate_frame(blurred=True)
    artificial = np.random.default_rng(92).integers(0, 256, (110, 390, 3), np.uint8)
    with source_view(original, left=40, top=20, x_scale=.1, y_scale=.1):
        assert not needs_plate_work(artificial, (0, 0, 390, 110))
    clear, _ = plate_frame()
    assert needs_plate_work(clear, box)


def test_unknown_cuda_pixels_do_not_trigger_a_new_download_or_suppress_confirmation():
    from vision_shield.detection.plate_readability import needs_plate_work

    class CUDA:
        is_cuda = True

        def cpu(self):
            raise AssertionError('No extra candidate download')

    assert needs_plate_work(CUDA(), (10, 10, 30, 20))


def test_original_source_pixels_are_never_modified():
    from vision_shield.detection.plate_readability import needs_plate_work

    gray, box = plate_frame(blurred=True)
    before = gray.copy()
    gray.setflags(write=False)
    needs_plate_work(gray, box)
    np.testing.assert_array_equal(gray, before)


@pytest.mark.parametrize('blurred,score,confirmed', [(True, .2, False), (False, .2, True), (True, .9, False)])
def test_quality_omits_only_extra_confirmation_and_keeps_direct_masks(monkeypatch, blurred, score, confirmed):
    from types import SimpleNamespace
    from vision_shield.detection import Detection
    from vision_shield.detection.quality import Detector

    gray, box = plate_frame(blurred=blurred)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    target = Detection('plate', box, score)
    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = 'cpu', 0
    detector.primary = SimpleNamespace(candidates=lambda *a, **kw: [target])
    detector.detail = SimpleNamespace(candidates=lambda *a, **kw: [])
    calls = []
    monkeypatch.setattr(detector, '_confirmed', lambda *a: calls.append(a) or True)
    assert detector.detect_regions(rgb) == ([target] if score >= .7 or confirmed else [])
    assert bool(calls) is confirmed


def test_proposal_uses_current_original_gray_and_restores_small_clear_plate_immediately():
    from vision_shield.detection import Detection
    from vision_shield.detection.plate_readability import needs_plate_work
    from vision_shield.redaction import FrameRedactor

    blurred, box = plate_frame(blurred=True)
    clear, _ = plate_frame()

    class Proposals:
        visible = False
        plate_needs_work = staticmethod(needs_plate_work)

        def detect_regions(self, pixels):
            return [Detection('plate', box, .2)] if self.visible else []

        def plate_proposals(self, pixels):
            return [Detection('plate', box, .2)]

    class CUDA:
        shape = (80, 160, 3)

    detector = Proposals()
    redactor = FrameRedactor(detector, detection_interval=3, tracking='flow', debug=True)
    pixels, thumbnail = CUDA(), np.zeros((36, 64, 3), np.float32)
    assert redactor.regions(pixels, 0, thumbnail=thumbnail, gray=blurred) == []
    assert redactor.regions(pixels, .04, thumbnail=thumbnail, gray=blurred) == []
    assert redactor.detection_stats['full_rechecks'] == 0
    detector.visible = True
    assert redactor.regions(pixels, .08, thumbnail=thumbnail, gray=clear) == [box]
    assert redactor.detection_stats['proposal_rechecks'] == 1


def test_nested_local_quality_confirmation_keeps_original_coordinates(monkeypatch):
    from types import SimpleNamespace
    from vision_shield.detection import Detection
    from vision_shield.detection.quality import Detector
    from vision_shield.redaction import FrameRedactor

    gray, box = plate_frame(blurred=True)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = 'cpu', 0

    def candidates(view, *, kind=None):
        # The source crop is [1:118, 0:80]; map its resized candidate back to the original.
        xs, ys = 117 / view.shape[1], 80 / view.shape[0]
        return [Detection('plate', ((box[0] - 1) / xs, box[1] / ys,
                                   (box[2] - 1) / xs, box[3] / ys), .2)]

    detector.primary = SimpleNamespace(candidates=candidates)
    detector.detail = SimpleNamespace(candidates=lambda *a, **kw: [])
    monkeypatch.setattr(detector, '_confirmed', lambda *a: pytest.fail('Blurred original must not be reconfirmed'))
    assert FrameRedactor(detector)._recheck(rgb, box, 'plate') is None
