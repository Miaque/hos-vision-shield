import os
import subprocess
import threading

import av
import numpy as np
import pytest
import torch

from tests.conftest import NO_WINDOW
from vision_shield.config import Settings
from vision_shield.media.video import process_video
from tests.integration.test_video_pipeline import FixedDetector


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVDEC/NVENC")
@pytest.mark.parametrize('tracking', ['template', 'flow'])
def test_every_third_gpu_detection_preserves_all_frames_clocks_and_audio(generated_media, tmp_path, tracking):
    from tests.integration.test_media_probe import audio_payloads

    class NoTargets:
        calls = 0

        def detect(self, pixels):
            assert pixels.is_cuda
            self.calls += 1
            return []

    source, target = generated_media / 'aac.mp4', tmp_path / 'cadence.mp4'
    detector, events = NoTargets(), []
    config = {**Settings(_env_file=None, inference_device='cuda').model_dump(), 'detection_interval': 3, 'tracking': tracking}
    process_video(source, target, detector, config, threading.Event(), events.append)
    with av.open(source) as reader:
        expected = [f.pts * f.time_base for f in reader.decode(video=0)]
    with av.open(target) as reader:
        assert [f.pts * f.time_base for f in reader.decode(video=0)] == expected
    assert len(expected) == 24
    counts = events[-1]['detection_stats']
    assert counts['local_rechecks'] == 0 and counts['full_frames'] == detector.calls
    assert counts['skipped_frames'] == len(expected) - detector.calls
    assert detector.calls == 8 + counts.get('full_rechecks', 0)
    assert events[-1]['video_decoder'] == 'h264_nvdec' and events[-1]['video_encoder'] == 'h264_nvenc'
    assert audio_payloads(source) == audio_payloads(target)


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVDEC/NVENC")
def test_failed_flow_rechecks_cuda_frame_before_masking(generated_media, tmp_path):
    class Target:
        def detect(self, pixels):
            assert pixels.is_cuda
            return [(50, 50, 100, 100)]

    events = []
    config = {**Settings(_env_file=None, inference_device='cuda').model_dump(), 'detection_interval': 3, 'tracking': 'flow'}
    source, target = tmp_path / 'flat-target.mp4', tmp_path / 'tracked.mp4'
    # A featureless target forces real LK failure; the moving test pattern is not a stationary target.
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(generated_media / 'aac.mp4'),
                    '-vf', 'drawbox=x=40:y=40:w=70:h=70:color=gray:t=fill', '-c:v', 'libx264',
                    '-c:a', 'copy', str(source)], check=True, capture_output=True, creationflags=NO_WINDOW)
    process_video(source, target, Target(), config, threading.Event(), events.append)
    with av.open(target) as reader:
        frames = list(reader.decode(video=0))
    assert len(frames) == 24
    assert all((np.abs(f.to_ndarray(format='rgb24')[55:95, 55:95].astype(int) - 64) < 10).all() for f in frames)
    counts = events[-1]['detection_stats']
    assert counts['flow_updates'] + counts['flow_failures'] > 0
    assert counts['flow_failures'] > 0 and counts['full_rechecks'] > 0
    assert counts['full_frames'] == 8 + counts['full_rechecks']


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVDEC/NVENC")
@pytest.mark.parametrize("angle", [90, 180, 270])
def test_nvdec_rotation_keeps_clocks_and_masks(generated_media, tmp_path, angle):
    source, target = tmp_path / "rotated.mp4", tmp_path / "result.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(angle), "-i", str(generated_media / "silent.mp4"),
                    "-c", "copy", str(source)], check=True, capture_output=True, creationflags=NO_WINDOW)
    events = []
    process_video(source, target, FixedDetector(), Settings(_env_file=None, inference_device="cuda").model_dump(),
                  threading.Event(), events.append)
    with av.open(source) as reader:
        expected = [frame.pts * frame.time_base for frame in reader.decode(video=0)]
    with av.open(target) as reader:
        frames = list(reader.decode(video=0))
    assert [frame.pts * frame.time_base for frame in frames] == expected
    dimensions = (240, 320) if angle != 180 else (320, 240)
    assert all((f.width, f.height, f.rotation) == (*dimensions, 0) for f in frames)
    assert all(abs(int(value) - 64) < 10 for value in frames[0].to_ndarray(format="rgb24")[20, 20])
    assert events[-1]["video_decoder"] == "h264_nvdec"


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVENC")
def test_unsupported_nvdec_pixel_format_keeps_existing_software_decode(generated_media, tmp_path):
    source, target = tmp_path / "444.mp4", tmp_path / "result.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(generated_media / "silent.mp4"),
                    "-c:v", "libx264", "-pix_fmt", "yuv444p", str(source)],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    events = []
    process_video(source, target, FixedDetector(), Settings(_env_file=None, inference_device="cuda").model_dump(),
                  threading.Event(), events.append)
    with av.open(target) as reader:
        assert len(list(reader.decode(video=0))) == 24
    assert events[-1]["video_decoder"] == "h264" and events[-1]["video_encoder"] == "h264_nvenc"


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVDEC/NVENC")
@pytest.mark.parametrize("mode", ["solid", "mosaic"])
def test_supported_frames_never_use_cpu_rgb_conversion_or_masking(generated_media, tmp_path, monkeypatch, mode):
    import cv2
    from vision_shield import redaction

    class CUDAOnly:
        def detect(self, pixels):
            assert pixels.is_cuda and pixels.dtype == torch.uint8
            return [(10, 10, 30, 30)]

    def forbidden(*args, **kwargs):
        pytest.fail("NVDEC frames must stay on GPU through conversion, masking and NVENC input")

    monkeypatch.setattr(cv2, "cvtColor", forbidden)
    monkeypatch.setattr(cv2, "resize", forbidden)
    monkeypatch.setattr(redaction, "redact", forbidden)
    config = Settings(_env_file=None, inference_device="cuda").model_dump()
    config["redaction_mode"] = mode
    events = []
    process_video(generated_media / "silent.mp4", tmp_path / "gpu.mp4", CUDAOnly(), config, threading.Event(), events.append)
    assert events[-1]["video_decoder"] == "h264_nvdec" and events[-1]["video_encoder"] == "h264_nvenc"


def test_nvdec_runtime_failure_is_not_retried_with_cpu(generated_media, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vision_shield.media import cuda_frames, video

    open_media, attempts = av.open, []

    def open_once(source, **kwargs):
        attempts.append(kwargs)
        if "hwaccel" in kwargs:
            raise RuntimeError("NVDEC unavailable")
        return open_media(source, **kwargs)

    monkeypatch.setattr(cuda_frames, "CUDAFrames", lambda: SimpleNamespace(device=0))
    monkeypatch.setattr(video.av, "open", open_once)
    with pytest.raises(RuntimeError, match="NVDEC unavailable"):
        process_video(generated_media / "silent.mp4", tmp_path / "result.mp4", FixedDetector(),
                      Settings(_env_file=None, inference_device="cuda").model_dump(), threading.Event(), lambda e: None)
    assert len(attempts) == 2 and "hwaccel" not in attempts[0] and "hwaccel" in attempts[1]


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual CUDA models and NVDEC")
@pytest.mark.parametrize("precision", ["automatic", "half", "half-flow"])
@pytest.mark.parametrize("filename,start,end", [("人脸视频1.mp4", 79.5, 80.5), ("人脸视频2.mp4", 9.8, 10.5)])
def test_nvdec_keeps_real_moving_tracks_and_gray_references(filename, start, end, precision):
    from pathlib import Path
    from vision_shield.detection import load_detector
    from vision_shield.detection.egoblur import _HalfPlate
    from vision_shield.media.cuda_frames import CUDAFrames, hwaccel, rgb_pixels
    from vision_shield.redaction import FrameRedactor

    detector = load_detector(Settings(_env_file=None, inference_device="cuda", detection_backend="quality_yolo").model_dump())
    if precision.startswith("half") and not isinstance(detector.primary.models[1], _HalfPlate):
        detector.primary.models[1] = _HalfPlate(detector.primary.models[1])
    frames = CUDAFrames()
    options = dict(detection_interval=3, tracking='flow') if precision == 'half-flow' else {}
    old, new = FrameRedactor(detector, **options), FrameRedactor(detector, **options)
    source = Path("tests/test_assets/report") / filename
    with av.open(source) as software, av.open(source, hwaccel=hwaccel(frames.device)) as hardware:
        for reader in (software, hardware):
            stream = reader.streams.video[0]
            reader.seek(int(start / stream.time_base), stream=stream)
        references = iter(software.decode(video=0))
        count = 0
        for frame in hardware.decode(video=0):
            reference = next(references)
            assert frame.pts * frame.time_base == reference.pts * reference.time_base
            moment = float(frame.pts * frame.time_base)
            if moment < start:
                continue
            if moment > end:
                break
            pixels = reference.to_ndarray(format="rgb24")
            masked = old.apply(pixels, moment)
            output = frames.masked(frame, new)
            assert output.format.name == "cuda" and output.sw_format.name == "nv12"
            np.testing.assert_array_equal(old.gray, new.gray)
            assert len(old.history) == len(new.history)
            assert [track.kind for track in old.history] == [track.kind for track in new.history]
            np.testing.assert_allclose([track.box for track in old.history], [track.box for track in new.history], rtol=0, atol=0.001)
            if filename == "人脸视频1.mp4" and moment == 80:
                planes = [torch.from_dlpack(plane.__dlpack__()) for plane in output.planes]
                actual = rgb_pixels(*planes, output.colorspace, output.color_range == 2)
                for x, y, a, b in [(1658, 217, 1676, 239), (1610, 234, 1625, 252), (1778, 244, 1793, 260)]:
                    assert (masked[y:b, x:a] == 64).all()
                    assert ((actual[y:b, x:a].int() - 64).abs() <= 2).all()
                assert np.array_equal(masked[210:245, 1556:1587], pixels[210:245, 1556:1587])
                assert np.array_equal(masked[299:338, 1612:1640], pixels[299:338, 1612:1640])
            if filename == "人脸视频2.mp4" and moment == 10:
                planes = [torch.from_dlpack(plane.__dlpack__()) for plane in output.planes]
                actual = rgb_pixels(*planes, output.colorspace, output.color_range == 2)
                assert ((actual[182:204, 393:400].int() - 64).abs() <= 2).all()
                np.testing.assert_array_equal(actual[480:520, 380:490].cpu().numpy(), pixels[480:520, 380:490])
            count += 1
    assert count >= 15
    for name in ("color-convert", "gray-copy", "mask-gpu", "explicit-sync"):
        assert detector.stage_times.snapshot()[name] > 0, name


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual CUDA models and NVDEC")
def test_flow_proposals_recover_entry_growth_and_profile_expiry():
    from pathlib import Path
    from vision_shield.detection import load_detector
    from vision_shield.detection.egoblur import _HalfPlate
    from vision_shield.media.cuda_frames import CUDAFrames, hwaccel, rgb_pixels
    from vision_shield.redaction import FrameRedactor

    detector = load_detector(Settings(_env_file=None, inference_device="cuda", detection_backend="quality_yolo").model_dump())
    if not isinstance(detector.primary.models[1], _HalfPlate):
        detector.primary.models[1] = _HalfPlate(detector.primary.models[1])
    cases = [("22-27-35车撞人.mp4", 42, 44.166, (1490, 505, 1530, 560)),
             ("22-27-35车撞人.mp4", 50.8, 52.273, (90, 85, 215, 140)),
             ("report/人脸视频1.mp4", 118.5, 120, (16, 214, 36, 244)),
             ("report/人脸视频1.mp4", 102.5, 103.25, (1809, 202, 1828, 213))]
    for filename, start, target, box in cases:
        frames = CUDAFrames()
        baseline = FrameRedactor(detector)
        candidate = FrameRedactor(detector, detection_interval=3, tracking="flow")
        checked = False
        with av.open(Path("tests/test_assets") / filename, hwaccel=hwaccel(frames.device)) as reader:
            for index, frame in enumerate(reader.decode(video=0)):
                moment = float(frame.pts * frame.time_base)
                if moment < start:
                    continue
                if moment > target + 1e-6:
                    break
                if abs(moment - target) < 1e-6:
                    planes = [torch.from_dlpack(plane.__dlpack__()) for plane in frame.planes]
                    pixels = rgb_pixels(*planes, frame.colorspace, frame.color_range == 2)
                    joint = sorted((d for d in detector.detect_regions(pixels) if d.kind == "face"), key=lambda d: d.box)
                    faces = sorted(detector.detect_faces(pixels), key=lambda d: d.box)
                    assert all(d.kind == "face" for d in faces)
                    np.testing.assert_allclose([(*d.box, d.score) for d in faces],
                                               [(*d.box, d.score) for d in joint], rtol=0, atol=.001)
                    expected = sorted(detector.detect_regions(pixels), key=lambda d: (d.kind, d.box))
                    detector.plate_proposals(pixels)
                    reused = sorted(detector.detect_regions(pixels), key=lambda d: (d.kind, d.box))
                    assert [d.kind for d in reused] == [d.kind for d in expected]
                    np.testing.assert_allclose([(*d.box, d.score) for d in reused],
                                               [(*d.box, d.score) for d in expected], rtol=0, atol=.001)
                    detector.plate_proposals(pixels)
                    reused = sorted(detector.detect_faces(pixels), key=lambda d: d.box)
                    np.testing.assert_allclose([(*d.box, d.score) for d in reused],
                                               [(*d.box, d.score) for d in joint], rtol=0, atol=.001)
                candidate.frame_index = index
                for redactor in (baseline, candidate):
                    output = frames.masked(frame, redactor)
                    if abs(moment - target) < 1e-6:
                        planes = [torch.from_dlpack(plane.__dlpack__()) for plane in output.planes]
                        actual = rgb_pixels(*planes, output.colorspace, output.color_range == 2)
                        x, y, a, b = box
                        assert ((actual[y:b, x:a].int() - 64).abs() <= 2).all(), (filename, moment, redactor.tracking)
                        assert output.pts * output.time_base == frame.pts * frame.time_base
                        checked = True
        assert checked
        assert candidate.detection_stats["face_rechecks"] > 0
    assert detector.stage_times.snapshot()["track-probe"] > 0
