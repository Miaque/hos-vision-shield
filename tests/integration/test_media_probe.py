"""T007: round-trip real compressed media and verify clocks and AAC payloads."""

from fractions import Fraction
import json
import os
import shutil
import subprocess

import av
import pytest

pytestmark = pytest.mark.integration


def audio_payloads(path):
    with av.open(str(path)) as container:
        return [bytes(packet) for packet in container.demux(audio=0) if packet.size]


@pytest.mark.parametrize("name", ["silent.mp4", "aac.mp4", "vfr.mp4"])
@pytest.mark.parametrize("drop_old_frames", [False, True])
def test_timestamp_and_audio_roundtrip(generated_media, tmp_path, name, drop_old_frames):
    source = generated_media / name
    target = tmp_path / name
    with av.open(str(source)) as reader, av.open(str(target), "w") as writer:
        video_in = reader.streams.video[0]
        video_out = writer.add_stream("libx264", rate=video_in.average_rate)
        video_out.width, video_out.height = video_in.width, video_in.height
        video_out.pix_fmt = "yuv420p"
        video_out.time_base = video_in.time_base
        video_out.codec_context.time_base = video_in.time_base
        audio_out = writer.add_stream_from_template(reader.streams.audio[0]) if reader.streams.audio else None
        input_times = []
        seen = 0
        for packet in reader.demux():
            if packet.stream.type == "audio" and packet.dts is not None:
                packet.stream = audio_out
                writer.mux(packet)
            elif packet.stream.type == "video":
                for frame in packet.decode():
                    seen += 1
                    if drop_old_frames and seen % 2 == 0:
                        continue
                    input_times.append(frame.pts * frame.time_base)
                    for encoded in video_out.encode(frame):
                        writer.mux(encoded)
        for packet in video_out.encode():
            writer.mux(packet)
    with av.open(str(target)) as result:
        output_times = [frame.pts * frame.time_base for frame in result.decode(video=0)]
        assert len(output_times) == len(input_times) == (12 if drop_old_frames else 24)
        assert all(abs(a - b) <= Fraction(1, 1000) for a, b in zip(input_times, output_times, strict=True))
    if name == "aac.mp4":
        assert audio_payloads(source) == audio_payloads(target)
    binary = os.getenv("TEST_FFPROBE") or shutil.which("ffprobe")
    assert binary, "FFprobe is required to independently validate media output"
    result = subprocess.run(
        [binary, "-v", "error", "-show_streams", "-count_frames", "-of", "json", str(target)],
        check=True, capture_output=True, timeout=15,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert video["codec_name"] == "h264"
    assert int(video["nb_read_frames"]) == len(input_times)
    assert [stream["codec_name"] for stream in streams if stream["codec_type"] == "audio"] == (
        ["aac"] if name == "aac.mp4" else []
    )
