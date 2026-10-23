"""Finite media processing: preserve source clocks and compressed AAC payloads."""

from fractions import Fraction
from contextlib import ExitStack
import time
from pathlib import Path
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING

import av
import numpy as np

from vision_shield.redaction import redact, FrameRedactor
from vision_shield.detection.timing import snapshot
if TYPE_CHECKING:
    from vision_shield.detection import Detector


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


def add_video(writer, frame, source, live=False, *, device="cpu"):
    encoder = "h264_nvenc" if device == "cuda" and not live else "libx264"
    hardware = getattr(getattr(frame, "format", None), "name", None) == "cuda"
    options = {}
    if hardware:
        import torch
        from vision_shield.media.cuda_frames import hwaccel

        options["hwaccel"] = hwaccel(torch.cuda.current_device())
    video = writer.add_stream(encoder, rate=source.average_rate or 25, **options)
    video.width, video.height = frame.width, frame.height
    if frame.width % 2 or frame.height % 2:
        raise ValueError("H.264 output requires even dimensions")
    video.pix_fmt = "cuda" if hardware else "yuv420p"
    if hardware:
        for name in ("colorspace", "color_range", "color_trc", "color_primaries"):
            setattr(video.codec_context, name, getattr(frame, name))
    video.time_base = frame.time_base
    video.codec_context.time_base = frame.time_base
    if live:
        video.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
        # Sparse processed frames must remain immediately decodable by new readers.
        video.codec_context.gop_size = 1
    else:
        if encoder == "h264_nvenc":
            import torch
            video.bit_rate = 0
            video.options = {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": "19",
                             "gpu": str(torch.cuda.current_device()), "bf": "3"}
        else:
            video.options = {"preset": "slow", "crf": "23"}
        if source.bit_rate and source.bit_rate > 0:
            # 保留质量目标，以源平均码率的 115% 设两秒 VBV；不是文件大小硬上限。
            maximum = round(source.bit_rate * 1.15)
            video.options.update(maxrate=str(maximum), bufsize=str(maximum * 2))
    return video


def process_video(source: str | Path, target: str | Path, detector: "Detector", config: dict,
                  stop: Event, emit: Callable[[dict], None]) -> int:
    """Write complete MP4, preserving frame clocks and AAC packets; return frame count."""
    count, started = 0, time.monotonic()
    frame_seconds = encode_seconds = 0.0
    redactor = FrameRedactor(detector, mode=config.get("redaction_mode", "solid"),
                             detection_interval=config.get("detection_interval", 1), tracking=config.get("tracking", "template"))
    with ExitStack() as stack:
        reader = stack.enter_context(av.open(source))
        brand = reader.metadata.get("major_brand", "").lower().strip()
        if "mp4" not in reader.format.name.split(",") or brand.startswith(("qt", "3g", "mjp", "mj2")):
            raise ValueError("MP4 container required")
        input_video = check_streams(reader, config)
        if reader.duration and reader.duration / av.time_base > config["max_video_seconds"]:
            raise ValueError("Video duration exceeds limit")
        frames = None
        if (config.get("inference_device") == "cuda" and input_video.codec_context.pix_fmt in {"yuv420p", "yuvj420p"}
                and input_video.codec_context.colorspace in {0, 1, 2, 3, 5, 6}):
            from vision_shield.media.cuda_frames import CUDAFrames, hwaccel

            frames = CUDAFrames()
            reader = stack.enter_context(av.open(source, hwaccel=hwaccel(frames.device)))
            input_video = check_streams(reader, config)
        writer = stack.enter_context(av.open(target, "w", format="mp4"))
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
                    stage = time.monotonic()
                    output = frames.masked(frame, redactor) if frames else masked_frame(frame, redactor)
                    frame_seconds += time.monotonic() - stage
                    stage = time.monotonic()
                    if video is None:
                        video = add_video(writer, output, input_video, device=config.get("inference_device", "cpu"))
                        for audio_packet in pending_audio:
                            writer.mux(audio_packet)
                        pending_audio.clear()
                    if (output.width, output.height) != (video.width, video.height):
                        raise ValueError("Video dimensions changed")
                    for encoded in video.encode(output):
                        writer.mux(encoded)
                    encode_seconds += time.monotonic() - stage
                    count += 1
                    emit({"video_decoder": "h264_nvdec" if frames else "h264",
                          "video_encoder": video.codec_context.name,
                          "detection_stats": redactor.detection_stats.copy(),
                          "stats": {"processed_frames": count, "dropped_frames": 0,
                                    "elapsed_seconds": time.monotonic() - started},
                          "timings": dict(frame=frame_seconds, inference=redactor.inference_seconds,
                                          encode=encode_seconds, **redactor.candidate_timings, **snapshot(detector))})
        if not video or not count:
            raise ValueError("No valid video frames")
        stage = time.monotonic()
        for packet in video.encode():
            writer.mux(packet)
    encode_seconds += time.monotonic() - stage
    emit({"video_decoder": "h264_nvdec" if frames else "h264",
          "video_encoder": video.codec_context.name,
          "detection_stats": redactor.detection_stats.copy(),
          "stats": {"processed_frames": count, "dropped_frames": 0, "elapsed_seconds": time.monotonic() - started},
          "timings": dict(frame=frame_seconds, inference=redactor.inference_seconds,
                          encode=encode_seconds, **redactor.candidate_timings, **snapshot(detector))})
    return count
