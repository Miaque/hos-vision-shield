import av
import cv2
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("space,full", [(1, False), (1, True), (2, False), (2, True)])
def test_device_nv12_conversion_matches_software_decoder_pixels(space, full):
    from vision_shield.media.cuda_frames import rgb_pixels

    rng = np.random.default_rng(91)
    y = rng.integers(0, 256, (64, 256), np.uint8)
    uv = rng.integers(0, 256, (32, 128, 2), np.uint8)
    planar = np.concatenate((y.ravel(), uv[:, :, 0].ravel(), uv[:, :, 1].ravel())).reshape(96, 256)
    frame = av.VideoFrame.from_ndarray(planar, format="yuv420p")
    frame.colorspace, frame.color_range = space, 2 if full else 1
    actual = rgb_pixels(torch.from_numpy(y), torch.from_numpy(uv), space, full).numpy()
    np.testing.assert_array_equal(actual, frame.to_ndarray(format="rgb24"))


def test_device_gray_keeps_the_existing_template_matching_pixels():
    from vision_shield.media.cuda_frames import gray_pixels

    pixels = np.random.default_rng(48).integers(0, 256, (128, 256, 3), np.uint8)
    np.testing.assert_array_equal(gray_pixels(torch.from_numpy(pixels)).numpy(), cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("mode", ["solid", "mosaic"])
def test_device_masks_preserve_decoder_references_and_unmasked_yuv(full, mode):
    from vision_shield.media.cuda_frames import masked_planes, rgb_pixels

    y = torch.arange(64 * 128, dtype=torch.int32).reshape(64, 128).remainder(220).byte()
    uv = torch.full((32, 64, 2), 128, dtype=torch.uint8)
    original_y, original_uv = y.clone(), uv.clone()
    pixels = rgb_pixels(y, uv, 2, full)
    masked_y, masked_uv = masked_planes(y, uv, pixels, [(10, 12, 40, 42)], mode, 2, full)
    assert torch.equal(y, original_y) and torch.equal(uv, original_uv)
    assert torch.equal(masked_y[:12], y[:12]) and torch.equal(masked_uv[:6], uv[:6])
    actual = rgb_pixels(masked_y, masked_uv, 2, full)
    if mode == "solid":
        assert torch.all((actual[16:38, 14:36].int() - 64).abs() <= 2)
    else:
        assert torch.unique(actual[16:38, 14:36, 0]).numel() <= 4
        assert not torch.all((actual[16:38, 14:36].int() - 64).abs() <= 2)
    assert masked_uv.data_ptr() - masked_y.data_ptr() == masked_y.stride(0) * masked_y.shape[0]


def test_invalid_device_mask_coordinates_are_rejected():
    from vision_shield.media.cuda_frames import masked_planes

    with pytest.raises(ValueError, match="Non-finite"):
        masked_planes(torch.zeros((8, 8), dtype=torch.uint8), torch.zeros((4, 4, 2), dtype=torch.uint8),
                      torch.zeros((8, 8, 3), dtype=torch.uint8), [(float("nan"), 1, 4, 4)], "solid", 2, False)


@pytest.mark.parametrize("space,full,message", [(9, 1, "color matrix"), (1, 1, "properties changed"),
                                               (2, 2, "properties changed")])
def test_device_frame_color_changes_cannot_silently_use_wrong_encoder_metadata(space, full, message):
    from types import SimpleNamespace
    from vision_shield.media.cuda_frames import CUDAFrames

    processor = CUDAFrames.__new__(CUDAFrames)
    processor.color = (False, False)
    frame = SimpleNamespace(format=SimpleNamespace(name="cuda"), sw_format=SimpleNamespace(name="nv12"),
                            rotation=0, colorspace=space, color_range=full)
    with pytest.raises(ValueError, match=message):
        processor.masked(frame, None)
