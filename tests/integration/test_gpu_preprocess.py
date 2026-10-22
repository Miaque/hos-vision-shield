"""Actual CUDA inputs must preserve the existing four model pixel contracts."""

import os

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from vision_shield.config import Settings
from vision_shield.detection import load_detector, cuda_pixels


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual CUDA models")
@pytest.mark.parametrize("height,width", [(321, 319), (1080, 3840)])
def test_gpu_preprocessing_binds_matching_pixels_without_cpu_image_operations(monkeypatch, height, width):
    detector = load_detector(Settings(_env_file=None, inference_device="cuda", detection_backend="quality_yolo").model_dump())
    primary, detail = detector.primary, detector.detail
    pixels = np.random.default_rng(47).integers(0, 256, (height, width, 3), dtype=np.uint8)
    side = max(32, ((min(1280, max(height, width)) + 31) // 32) * 32)
    scale = side / max(height, width)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    dx, dy = (side - rw) % 32, (side - rh) % 32
    view = cv2.copyMakeBorder(cv2.resize(pixels, (rw, rh)), round(dy / 2 - .1), round(dy / 2 + .1),
                             round(dx / 2 - .1), round(dx / 2 + .1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    face_session = primary._session(*view.shape[:2])
    expected = {face_session: [(cv2.dnn.blobFromImage(view, 1 / 255), 1 / 255 + 1e-6)],
                detail.face: [], detail.plate: []}
    scale = min(1, 1280 / max(height, width))
    plate = np.asarray(Image.fromarray(pixels).resize((max(1, round(width * scale)), max(1, round(height * scale))),
                                                    Image.Resampling.LANCZOS)) if scale < 1 else pixels
    expected_plate = plate[:, :, ::-1].transpose(2, 0, 1)
    for left, top, right, bottom in detail._views(width, height, detail.tile_side):
        view = cv2.cvtColor(pixels[top:bottom, left:right], cv2.COLOR_RGB2BGR)
        vh, vw = view.shape[:2]
        scale = min(1, detail.max_side / max(vw, vh))
        if scale < 1:
            view = cv2.resize(view, (max(1, round(vw * scale)), max(1, round(vh * scale))), interpolation=cv2.INTER_AREA)
        vh, vw = view.shape[:2]
        padded = cv2.copyMakeBorder(view, 0, (-vh) % 32, 0, (-vw) % 32, cv2.BORDER_CONSTANT)
        expected[detail.face].append((cv2.dnn.blobFromImage(padded), 1))
        scale = min(320 / vw, 320 / vh)
        rw, rh = max(1, round(vw * scale)), max(1, round(vh * scale))
        left, top = (320 - rw) // 2, (320 - rh) // 2
        canvas = np.zeros((320, 320, 3), np.uint8)
        canvas[top:top + rh, left:left + rw] = cv2.resize(view, (rw, rh))
        # The area and linear passes may each round by one in the composed detail path.
        expected[detail.plate].append((cv2.dnn.blobFromImage(canvas, 1 / 255, swapRB=True), 2 / 255 + 1e-6))
    run = cuda_pixels.run

    def capture(session, tensor, stream):
        assert tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous()
        reference_session = detail.face if session in detail.face_sessions.values() else session
        reference, tolerance = expected[reference_session].pop(0)
        np.testing.assert_allclose(tensor.cpu().numpy(), reference, rtol=0, atol=tolerance)
        return run(session, tensor, stream)

    model = primary.models[1]
    images = []

    def capture_plate(image):
        assert image.is_cuda and image.dtype == torch.uint8
        assert np.abs(image.cpu().numpy().astype(np.int16) - expected_plate).max() <= 1
        images.append(image.shape)
        return model(image)

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA model inputs must not return to CPU image operations")

    monkeypatch.setattr(cuda_pixels, "run", capture)
    monkeypatch.setattr(primary, "models", [None, capture_plate])
    for name in ("resize", "cvtColor"):
        monkeypatch.setattr(cv2, name, forbidden)
    monkeypatch.setattr(cv2.dnn, "blobFromImage", forbidden)
    tensor = primary.prepare(pixels)
    primary.candidates(tensor)
    detail.candidates(tensor)
    assert all(not remaining for remaining in expected.values()) and len(images) == 1
    np.testing.assert_array_equal(tensor.cpu().numpy(), pixels)
