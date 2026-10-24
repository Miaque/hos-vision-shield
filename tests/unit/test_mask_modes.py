import numpy as np
import pytest
from vision_shield.redaction import redact, FrameRedactor
from vision_shield.api.schemas import StreamInput

from pydantic import ValidationError


def test_default_is_opaque_dark_gray_and_preserves_input():
    pixels = np.full((24, 32, 3), 210, np.uint8)
    result = redact(pixels, [(-1, 2, 12, 20)])
    assert (result[2:20, :12] == 64).all()
    assert (result[:, 12:] == 210).all()
    assert (pixels == 210).all()


def test_mosaic_has_constant_blocks_and_preserves_outside():
    pixels = np.random.default_rng(5).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    result = redact(pixels, [(8, 8, 32, 32)], mode='mosaic')
    assert np.array_equal(result[:8], pixels[:8])
    assert not np.array_equal(result[8:32, 8:32], pixels[8:32, 8:32])
    assert np.unique(result[8:32, 8:32].reshape(-1, 3), axis=0).shape[0] <= 4
    assert (redact(pixels, [(0, 0, 1, 1)], mode='mosaic')[0, 0] == 64).all()


def test_mode_is_validated():
    assert StreamInput(url="rtsp://example.com/live").redaction_mode == 'solid'
    assert StreamInput(url="rtsp://example.com/live", redaction_mode='mosaic').redaction_mode == 'mosaic'
    with pytest.raises(ValidationError):
        StreamInput(url="rtsp://example.com/live", redaction_mode='blur')
    with pytest.raises(ValueError):
        redact(np.zeros((2, 2, 3), np.uint8), [], mode='blur')


def test_stream_bitrate_is_optional_positive_bits_per_second():
    assert StreamInput(url="rtsp://example.com/live").max_bitrate is None
    assert StreamInput(url="rtsp://example.com/live", max_bitrate=20_000_000).max_bitrate == 20_000_000
    with pytest.raises(ValidationError):
        StreamInput(url="rtsp://example.com/live", max_bitrate=0)


def test_image_and_temporal_paths_use_selected_mode(tmp_path):
    from PIL import Image
    from vision_shield.media.images import process_image
    class Detector:
        def detect(self, pixels): return [(0, 0, 24, 24)]
    pixels = np.random.default_rng(9).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    source, target = tmp_path/'source.png', tmp_path/'target.png'
    Image.fromarray(pixels).save(source)
    process_image(source, target, Detector(), 2000, redaction_mode='mosaic')
    expected = redact(pixels, Detector().detect(pixels), mode='mosaic')
    assert np.array_equal(np.array(Image.open(target)), expected)
    assert np.array_equal(FrameRedactor(Detector(), mode='mosaic').apply(pixels, 0), expected)
