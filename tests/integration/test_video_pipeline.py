import os
import threading
import subprocess

import av
import pytest

from vision_shield.config import Settings
from vision_shield.media.video import process_video, masked_frame, check_dimensions
from tests.integration.test_media_probe import audio_payloads
from tests.conftest import NO_WINDOW


class FixedDetector:
    def detect(self, pixels):
        return [(10, 10, 30, 30)]


@pytest.mark.parametrize("name", ["silent.mp4", "aac.mp4", "vfr.mp4"])
def test_video_preserves_frames_clocks_and_audio(generated_media, tmp_path, name):
    source, target = generated_media / name, tmp_path / name
    process_video(str(source), str(target), FixedDetector(), Settings(_env_file=None).model_dump(), threading.Event(), lambda event: None)
    with av.open(str(source)) as a:
        expected = [frame.pts * frame.time_base for frame in a.decode(video=0)]
    with av.open(str(target)) as b:
        frames = list(b.decode(video=0))
        assert [frame.pts * frame.time_base for frame in frames] == expected
        assert all(abs(int(v) - 64) < 10 for v in frames[0].to_ndarray(format="rgb24")[20, 20])
    if name == "aac.mp4":
        assert audio_payloads(source) == audio_payloads(target)


def test_rotation_is_applied_and_cancellation_stops_processing(generated_media, tmp_path):
    source = tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(generated_media / "silent.mp4"),
                    "-c", "copy", str(source)],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    target = tmp_path / "result.mp4"
    config = Settings(_env_file=None).model_dump()
    process_video(str(source), str(target), FixedDetector(), config, threading.Event(), lambda event: None)
    with av.open(str(target)) as reader:
        frames = list(reader.decode(video=0))
        assert len(frames) == 24
        assert (frames[0].width, frames[0].height, frames[0].rotation) == (240, 320, 0)
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        process_video(str(source), str(target), FixedDetector(), config, stop, lambda event: None)


@pytest.mark.parametrize("suffix", ["mkv", "3gp"])
def test_non_mp4_h264_container_is_rejected(generated_media, tmp_path, suffix):
    source = tmp_path / f"unsupported.{suffix}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(generated_media / "silent.mp4"),
                    "-c", "copy", str(source)], check=True, capture_output=True, creationflags=NO_WINDOW)
    with pytest.raises(ValueError, match="MP4"):
        process_video(str(source), str(tmp_path / "result.mp4"), FixedDetector(),
                      Settings(_env_file=None).model_dump(), threading.Event(), lambda event: None)


def test_missing_frame_clock_and_damaged_input_never_produce_result(tmp_path):
    with pytest.raises(ValueError, match="timestamp"):
        masked_frame(av.VideoFrame(320, 240, "rgb24"), FixedDetector())
    source, target = tmp_path / "broken.mp4", tmp_path / "result.mp4"
    source.write_bytes(b"damaged MP4")
    with pytest.raises(av.error.FFmpegError):
        process_video(str(source), str(target), FixedDetector(), Settings(_env_file=None).model_dump(),
                      threading.Event(), lambda event: None)
    assert not target.exists()


def test_decoded_frame_dimensions_are_bounded_independently_of_metadata():
    config = Settings(_env_file=None, max_video_width=320, max_video_height=240).model_dump()
    frame = av.VideoFrame(320, 240, "rgb24")
    check_dimensions(frame.width, frame.height, config)
    for width, height in [(640, 240), (320, 480)]:
        frame = av.VideoFrame(width, height, "rgb24")
        with pytest.raises(ValueError, match="dimensions"):
            check_dimensions(frame.width, frame.height, config)


def test_video_mosaic_configuration_reaches_encoded_frames(generated_media, tmp_path):
    from vision_shield.redaction import redact
    source, target = generated_media / 'silent.mp4', tmp_path / 'mosaic.mp4'
    config = {**Settings(_env_file=None).model_dump(), "redaction_mode": "mosaic"}
    process_video(source, target, FixedDetector(), config, threading.Event(), lambda event: None)
    with av.open(str(source)) as reader:
        pixels = next(reader.decode(video=0)).to_ndarray(format='rgb24')
    expected = redact(pixels, [(10, 10, 30, 30)], mode='mosaic')[20, 20]
    with av.open(str(target)) as reader:
        actual = next(reader.decode(video=0)).to_ndarray(format='rgb24')[20, 20]
    assert all(abs(int(a) - int(b)) < 10 for a, b in zip(actual, expected))
    assert any(abs(int(v) - 64) > 10 for v in actual), '必须实际使用马赛克而非默认灰块'


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual NVIDIA video encoder")
@pytest.mark.parametrize("name,mode", [("silent.mp4", "solid"), ("aac.mp4", "solid"),
                                       ("aac.mp4", "mosaic"), ("vfr.mp4", "solid")])
def test_nvenc_preserves_all_frames_clocks_audio_and_requested_mask(generated_media, tmp_path, name, mode):
    from vision_shield.redaction import redact

    source, target = generated_media / name, tmp_path / name
    config = {**Settings(_env_file=None).model_dump(), "inference_device": "cuda", "redaction_mode": mode}
    events = []
    process_video(source, target, FixedDetector(), config, threading.Event(), events.append)
    with av.open(str(source)) as reader:
        frames = list(reader.decode(video=0))
        clocks = [frame.pts * frame.time_base for frame in frames]
        expected = redact(frames[0].to_ndarray(format="rgb24"), [(10, 10, 30, 30)], mode=mode)[20, 20]
    with av.open(str(target)) as reader:
        frames = list(reader.decode(video=0))
        assert [frame.pts * frame.time_base for frame in frames] == clocks
        assert all((frame.width, frame.height) == (320, 240) for frame in frames)
        actual = frames[0].to_ndarray(format="rgb24")[20, 20]
        assert all(abs(int(a) - int(b)) < 10 for a, b in zip(actual, expected))
    assert events[-1]["stats"]["processed_frames"] == len(clocks)
    assert events[-1]["stats"]["dropped_frames"] == 0
    assert all(event["video_encoder"] == "h264_nvenc" for event in events)
    if name == "aac.mp4":
        assert audio_payloads(source) == audio_payloads(target)


def test_final_video_timing_includes_completed_container_and_keeps_frame_clocks(generated_media, tmp_path):
    import time

    target = tmp_path / "timed.mp4"
    events = []
    class SlowDetector:
        candidate_timings = {"candidate-primary": 0.0005, "candidate-detail": 0.0002}

        def detect(self, pixels):
            time.sleep(0.001)
            return []
    def observe(event):
        try:
            with av.open(str(target)) as reader:
                complete = len(list(reader.decode(video=0))) == 24
        except (av.error.FFmpegError, FileNotFoundError):
            complete = False
        events.append((event, complete))
    process_video(generated_media / "silent.mp4", target, SlowDetector(), Settings(_env_file=None).model_dump(),
                  threading.Event(), observe)
    final, complete = events[-1]
    assert complete, "最终计时必须在编码器冲刷和MP4封装关闭后发出"
    assert final["stats"]["processed_frames"] == 24 and final["stats"]["dropped_frames"] == 0
    assert set(final["stats"]) == {"processed_frames", "dropped_frames", "elapsed_seconds"}
    assert 0.024 <= final["timings"]["inference"] <= final["timings"]["frame"]
    assert final["timings"]["encode"] > 0
    assert final["timings"]["frame"] + final["timings"]["encode"] <= final["stats"]["elapsed_seconds"]
    assert final["timings"]["candidate-primary"] == pytest.approx(24 * 0.0005)
    assert final["timings"]["candidate-detail"] == pytest.approx(24 * 0.0002)
