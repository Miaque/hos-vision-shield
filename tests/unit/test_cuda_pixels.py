import cv2
import numpy as np
from PIL import Image
import pytest
import torch


@pytest.mark.parametrize("mode,shape,size", [
    (mode, shape, size) for mode in ("lanczos", "area", "linear")
    for shape, size in (((37, 63), (19, 11)), ((41, 59), (17, 23)), ((13, 17), (27, 31)))
    if mode != "area" or size[0] <= shape[1] and size[1] <= shape[0]
])
def test_model_resize_keeps_reference_interpolation(shape, size, mode):
    from vision_shield.detection.cuda_pixels import resize

    pixels = np.random.default_rng(123).integers(0, 256, (*shape, 3), dtype=np.uint8)
    original = pixels.copy()
    if mode == "lanczos":
        expected = np.array(Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS))
    else:
        expected = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA if mode == "area" else cv2.INTER_LINEAR)
    actual = resize(torch.from_numpy(pixels), size, mode).numpy()
    assert actual.dtype == np.uint8 and actual.shape == (*reversed(size), 3)
    assert np.abs(actual.astype(np.int16) - expected).max() <= 1
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("mode,size", [(mode, size) for mode in ("lanczos", "area")
                                     for size in ((7, 11), (15, 7), (15, 11), (5, 3), (27, 31))
                                     if mode != "area" or size[0] <= 15 and size[1] <= 11])
def test_strided_resize_keeps_axis_order_and_source(mode, size):
    from vision_shield.detection.cuda_pixels import resize

    source = np.random.default_rng(31).integers(0, 256, (21, 31, 3), dtype=np.uint8)
    original = source.copy()
    pixels = source[::2, 1::2]
    expected = (np.array(Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS)) if mode == "lanczos"
                else cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))
    actual = resize(torch.from_numpy(pixels), size, mode).numpy()
    assert actual.shape == expected.shape and actual.dtype == np.uint8
    assert np.abs(actual.astype(np.int16) - expected).max() <= 1
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("pixels", [np.zeros((0, 3, 3), np.uint8), np.zeros((2, 2, 4), np.uint8),
                                    np.zeros((2, 2, 3), np.float32), torch.zeros((2, 2, 3), dtype=torch.uint8)])
def test_invalid_cuda_pixels_fail_before_copy_or_inference(pixels):
    from vision_shield.detection.cuda_pixels import upload

    with pytest.raises(ValueError, match="RGB uint8"):
        upload(pixels, torch.device("cuda"))


def test_resize_coefficients_do_not_accumulate_for_changing_dimensions():
    from vision_shield.detection.cuda_pixels import _weights

    _weights.cache_clear()
    for width in range(20, 45):
        _weights(width, 10, "area", "cpu")
    assert _weights.cache_info().currsize == 16
    _weights.cache_clear()
