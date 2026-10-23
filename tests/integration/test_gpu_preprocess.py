"""Actual CUDA inputs must preserve the existing four model pixel contracts."""

import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from vision_shield.config import Settings
from vision_shield.detection import load_detector, cuda_pixels


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual CUDA resize")
@pytest.mark.parametrize("mode", ["lanczos", "area"])
def test_4k_resize_has_bounded_temporary_memory_and_matching_pixels(mode):
    pixels = np.random.default_rng(31).integers(0, 256, (1080, 3840, 3), dtype=np.uint8)
    tensor = torch.from_numpy(pixels).cuda()
    expected = (np.array(Image.fromarray(pixels).resize((1280, 360), Image.Resampling.LANCZOS)) if mode == "lanczos"
                else cv2.resize(pixels, (1280, 360), interpolation=cv2.INTER_AREA))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    actual = cuda_pixels.resize(tensor, (1280, 360), mode)
    torch.cuda.synchronize()
    # Bound per-view working memory while multiple detectors share the GPU.
    assert torch.cuda.max_memory_allocated() - initial < 96 * 1024**2
    assert np.abs(actual.cpu().numpy().astype(np.int16) - expected).max() <= 1
    np.testing.assert_array_equal(tensor.cpu().numpy(), pixels)


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

    def capture(session, tensor, stream, **timing):
        assert tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous()
        reference_session = (detail.face if session in detail.face_sessions.values() else
                             detail.plate if session in detail.plate_sessions.values() else session)
        references = [expected[reference_session].pop(0) for _ in range(tensor.shape[0])]
        reference = np.concatenate([pixels for pixels, _ in references])
        tolerance = max(value for _, value in references)
        np.testing.assert_allclose(tensor.cpu().numpy(), reference, rtol=0, atol=tolerance)
        return run(session, tensor, stream, **timing)

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


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual batched CUDA model")
def test_plate_batches_match_individual_inference_and_use_only_cuda(monkeypatch, tmp_path):
    import onnxruntime as ort
    from vision_shield.detection.cuda_detail import Detector

    create, sessions = ort.InferenceSession, []

    def profiled(model, *args, **kwargs):
        options = kwargs["sess_options"]
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / f"plate-{len(sessions)}")
        session = create(model, *args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(ort, "InferenceSession", profiled)
    config = Settings(_env_file=None)
    detector = Detector(config.face_model_path, config.plate_model_path)
    values = np.random.default_rng(27).random((4, 3, 320, 320), dtype=np.float32)
    expected = np.concatenate([detector.plate.run(None, {detector.plate.get_inputs()[0].name: value[None]})[0]
                               for value in values])
    with torch.cuda.stream(detector.plate_stream):
        tensor = torch.from_numpy(values).to(detector.device)
        for batch in (2, 3, 4):
            actual = detector._plate_forward(tensor[:batch])
            assert actual.shape == (batch, 6300, 15)
            np.testing.assert_allclose(actual, expected[:batch], rtol=1e-4, atol=1e-3)
    assert len(detector.plate_sessions) == 3
    for session in (detector.plate, *detector.plate_sessions.values()):
        events = json.loads(Path(session.end_profiling()).read_text("utf-8"))
        providers = {event["args"]["provider"] for event in events if "provider" in event.get("args", {})}
        assert providers == {"CUDAExecutionProvider"}, "每个批量都须验证实际算子，不能回退CPU"
