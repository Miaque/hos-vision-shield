"""One RTSP input, a bounded latest-frame buffer, and one output mux owner."""

from collections import deque
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import av

from vision_shield.media.video import add_video, check_streams, check_dimensions, masked_frame
from vision_shield.redaction import FrameRedactor
if TYPE_CHECKING:
    from vision_shield.detection import Detector


class StreamBuffer:
    def __init__(self, frame_limit, audio_seconds, audio_bytes, stop):
        self.frame_limit, self.audio_seconds, self.audio_limit = frame_limit, audio_seconds, audio_bytes
        self.stop, self.condition = stop, threading.Condition()
        self.frames, self.audio = deque(), deque()
        self.audio_size, self.dropped = 0, 0
        self.eof, self.error, self.input_video, self.input_audio = False, None, None, None

    def frame(self, value):
        with self.condition:
            # Some RTSP sources omit the first PTS; never publish or invent its clock.
            if value.pts is None or value.time_base is None:
                self.dropped += 1
                return
            if len(self.frames) == self.frame_limit:
                self.frames.popleft()
                self.dropped += 1
            self.frames.append(value)
            self.condition.notify()

    def audio_packet(self, packet):
        with self.condition:
            if packet.pts is None or packet.time_base is None:
                raise BufferError("Invalid audio clock")
            start = self.audio[0].pts * self.audio[0].time_base if self.audio else packet.pts * packet.time_base
            if self.audio_size + packet.size > self.audio_limit or packet.pts * packet.time_base - start > self.audio_seconds:
                raise BufferError("Audio backlog exceeded limit")
            self.audio.append(packet)
            self.audio_size += packet.size
            self.condition.notify()

    def take(self):
        with self.condition:
            while not self.frames and not self.eof and not self.error and not self.stop.is_set():
                self.condition.wait(0.2)
            if self.error:
                raise self.error
            if self.stop.is_set() or not self.frames:
                return None
            frame = self.frames.pop()
            self.dropped += len(self.frames)
            self.frames.clear()
            if frame.pts is None or frame.time_base is None:
                raise BufferError("Invalid video clock")
            moment = frame.pts * frame.time_base
            audio = []
            while self.audio and self.audio[0].pts * self.audio[0].time_base <= moment:
                packet = self.audio.popleft()
                self.audio_size -= packet.size
                audio.append(packet)
            return frame, audio


def process_stream(source: str, target: str, detector: "Detector", config: dict,
                   stop: threading.Event, emit: Callable[[dict], None]) -> None:
    """Read/publish RTSP until stop or failure; emit only processed frames and bounded stats."""
    buffer = StreamBuffer(config["frame_buffer_size"], config["audio_buffer_seconds"], config["audio_buffer_bytes"], stop)
    writer_done = threading.Event()
    redactor = FrameRedactor(detector, mode=config.get("redaction_mode", "solid"))

    def read_source():
        source_reader = None
        try:
            # RTSP SDP supplies codec parameters; long probing creates an audio burst.
            source_reader = av.open(source, options={"rtsp_transport": "tcp", "analyzeduration": "100000", "probesize": "32768"},
                                    timeout=(config["rtsp_connect_timeout"], config["rtsp_read_timeout"]))
            video = check_streams(source_reader, config)
            with buffer.condition:
                buffer.input_video = video
                buffer.input_audio = source_reader.streams.audio[0] if source_reader.streams.audio else None
            for packet in source_reader.demux():
                if stop.is_set():
                    break
                if packet.stream.type == "video":
                    for frame in packet.decode():
                        check_dimensions(frame.width, frame.height, config)
                        buffer.frame(frame)
                elif packet.stream.type == "audio" and packet.dts is not None:
                    buffer.audio_packet(packet)
        except Exception as error:
            with buffer.condition:
                buffer.error = error
        finally:
            with buffer.condition:
                buffer.eof = True
                buffer.condition.notify()
            # Output stream templates still refer to this container during mux setup.
            writer_done.wait()
            if source_reader:
                source_reader.close()

    reader = threading.Thread(target=read_source, daemon=True)
    reader.start()
    count, origin, previous = 0, None, None
    started = time.monotonic()
    last_output_report = None
    frame_seconds = encode_seconds = 0.0
    try:
        with av.open(target, "w", format="rtsp", options={"rtsp_transport": "tcp"},
                     timeout=(config["rtsp_connect_timeout"], config["rtsp_read_timeout"])) as writer:
            video, audio = None, None
            while sample := buffer.take():
                frame, packets = sample
                moment = frame.pts * frame.time_base
                if previous is not None and moment <= previous:
                    raise BufferError("Stream clock moved backwards")
                previous = moment
                if origin is None:
                    origin = moment
                stage = time.monotonic()
                output = masked_frame(frame, redactor, origin)
                frame_seconds += time.monotonic() - stage
                stage = time.monotonic()
                if video is None:
                    video = add_video(writer, output, buffer.input_video, live=True)
                    if buffer.input_audio:
                        audio = writer.add_stream_from_template(buffer.input_audio)
                if (output.width, output.height) != (video.width, video.height):
                    raise BufferError("Stream dimensions changed")
                for packet in packets:
                    packet.pts -= round(origin / packet.time_base)
                    packet.dts -= round(origin / packet.time_base)
                    packet.stream = audio
                    writer.mux(packet)
                published = False
                for packet in video.encode(output):
                    writer.mux(packet)
                    published = True
                encode_seconds += time.monotonic() - stage
                # 只有视频包实际写入成功才证明输出；合并到至多每五秒一次。
                if published and (last_output_report is None or time.monotonic() - last_output_report >= 5):
                    emit({"output": True, "output_monotonic": time.monotonic(), "ready": True})
                    last_output_report = time.monotonic()
                count += 1
                emit({"stats": {"processed_frames": count, "dropped_frames": buffer.dropped,
                                "elapsed_seconds": time.monotonic() - started},
                      "timings": dict(frame=frame_seconds, inference=redactor.inference_seconds,
                                      encode=encode_seconds, **redactor.candidate_timings)})
                if stop.is_set():
                    break
            if video:
                for packet in video.encode():
                    writer.mux(packet)
    finally:
        stop.set()
        writer_done.set()
        reader.join(timeout=config["rtsp_read_timeout"] + 1)
