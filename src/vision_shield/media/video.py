"""Finite media processing: preserve source clocks and compressed AAC payloads."""

from fractions import Fraction
import time
from pathlib import Path
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING

import av
import numpy as np

from vision_shield.redaction import redact, FrameRedactor
if TYPE_CHECKING:
    from vision_shield.detection.egoblur import Detector


def check_dimensions(width: int, height: int, config: dict) -> None:
    """Enforce the same bound on metadata and every decoded frame before RGB allocation."""
    if width > config["max_video_width"] or height > config["max_video_height"]:
        raise ValueError("Video dimensions exceed limit")


def check_streams(reader, config):
    if len(reader.streams.video) != 1 or len(reader.streams.audio) > 1:
        raise ValueError("Exactly one video and at most one audio track are supported")
    video = reader.streams.video[0]
    if video.codec_context.name != "h264":
        raise ValueError("H.264 required")
    check_dimensions(video.width, video.height, config)
    if reader.streams.audio and reader.streams.audio[0].codec_context.name != "aac":
        raise ValueError("AAC required")
    return video


def masked_frame(frame: av.VideoFrame, detector: "Detector | FrameRedactor", origin: Fraction = Fraction(0)) -> av.VideoFrame:
    if frame.pts is None or frame.time_base is None or frame.rotation % 90:
        raise ValueError("Unsupported video timestamp or orientation")
    pixels = frame.to_ndarray(format="rgb24")
    if frame.rotation:
        pixels = np.rot90(pixels, frame.rotation // 90).copy()
    masked = detector.apply(pixels, float(frame.pts * frame.time_base)) if isinstance(detector, FrameRedactor) else redact(pixels, detector.detect(pixels))
    output = av.VideoFrame.from_ndarray(masked, format="rgb24")
    output.time_base = frame.time_base
    output.pts = round((frame.pts * frame.time_base - origin) / frame.time_base)
    return output


def add_video(writer, frame, source, live=False):
    video = writer.add_stream("libx264", rate=source.average_rate or 25)
    video.width, video.height = frame.width, frame.height
    if frame.width % 2 or frame.height % 2:
        raise ValueError("H.264 output requires even dimensions")
    video.pix_fmt = "yuv420p"
    video.time_base = frame.time_base
    video.codec_context.time_base = frame.time_base
    if live:
        video.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
        # Sparse processed frames must remain immediately decodable by new readers.
        video.codec_context.gop_size = 1
    return video


def process_video(source: str | Path, target: str | Path, detector: "Detector", config: dict,
                  stop: Event, emit: Callable[[dict], None]) -> int:
    """Write complete MP4, preserving frame clocks and AAC packets; return frame count."""
    count, started = 0, time.monotonic()
    redactor = FrameRedactor(detector)
    with av.open(source) as reader, av.open(target, "w", format="mp4") as writer:
        brand = reader.metadata.get("major_brand", "").lower().strip()
        if "mp4" not in reader.format.name.split(",") or brand.startswith(("qt", "3g", "mjp", "mj2")):
            raise ValueError("MP4 container required")
        input_video = check_streams(reader, config)
        if reader.duration and reader.duration / av.time_base > config["max_video_seconds"]:
            raise ValueError("Video duration exceeds limit")
        audio = writer.add_stream_from_template(reader.streams.audio[0]) if reader.streams.audio else None
        video, pending_audio, pending_bytes, previous = None, [], 0, None
        for packet in reader.demux():
            if stop.is_set():
                raise InterruptedError("Processing cancelled")
            if packet.stream.type == "audio" and packet.dts is not None:
                packet.stream = audio
                if video is None:
                    pending_audio.append(packet)
                    pending_bytes += packet.size
                    if pending_bytes > config["audio_buffer_bytes"]:
                        raise ValueError("Video starts after excessive audio prefix")
                else:
                    writer.mux(packet)
            elif packet.stream.type == "video":
                for frame in packet.decode():
                    check_dimensions(frame.width, frame.height, config)
                    moment = frame.pts * frame.time_base if frame.pts is not None else None
                    if moment is None or (previous is not None and moment <= previous):
                        raise ValueError("Invalid source frame clock")
                    if previous is not None and moment - first > config["max_video_seconds"]:
                        raise ValueError("Video duration exceeds limit")
                    if previous is None:
                        first = moment
                    previous = moment
                    output = masked_frame(frame, redactor)
                    if video is None:
                        video = add_video(writer, output, input_video)
                        for audio_packet in pending_audio:
                            writer.mux(audio_packet)
                        pending_audio.clear()
                    if (output.width, output.height) != (video.width, video.height):
                        raise ValueError("Video dimensions changed")
                    for encoded in video.encode(output):
                        writer.mux(encoded)
                    count += 1
                    emit({"stats": {"processed_frames": count, "dropped_frames": 0,
                                    "elapsed_seconds": time.monotonic() - started}})
        if not video or not count:
            raise ValueError("No valid video frames")
        for packet in video.encode():
            writer.mux(packet)
    return count
