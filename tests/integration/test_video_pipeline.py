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
        assert max(frames[0].to_ndarray(format="rgb24")[20, 20]) < 10
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
