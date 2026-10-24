"""One RTSP input, bounded FIFO backpressure, and one output mux owner."""

from collections import deque
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import av
import httpx

from vision_shield.media.video import add_video, check_streams, check_dimensions, masked_frame
from vision_shield.media.stream_metrics import StreamMetrics
from vision_shield.redaction import FrameRedactor
from vision_shield.detection.timing import snapshot
if TYPE_CHECKING:
    from vision_shield.detection import Detector


class StreamBuffer:
    def __init__(self, frame_limit, audio_seconds, audio_bytes, stop):
        self.frame_limit, self.audio_seconds, self.audio_limit = frame_limit, audio_seconds, audio_bytes
        self.stop, self.condition = stop, threading.Condition()
        self.frames, self.audio = deque(), deque()
        self.received, self.last_received, self.input_frames = deque(), None, 0
        self.audio_size, self.dropped = 0, 0
        self.eof, self.error, self.input_video, self.input_audio = False, None, None, None

    def frame(self, value, received_at=None):
        with self.condition:
            self.input_frames += 1
            # Some RTSP sources omit the first PTS; never publish or invent its clock.
            if value.pts is None or value.time_base is None:
                self.dropped += 1
                return
            while len(self.frames) == self.frame_limit and not self.stop.is_set():
                self.condition.wait(0.2)
            if self.stop.is_set():
                return
            self.frames.append(value)
            self.received.append(received_at)
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
            frame = self.frames.popleft()
            self.last_received = self.received.popleft()
            self.condition.notify()
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
    """Read RTSP and publish RTSP/RTMP; emit only processed frames and bounded stats."""
    protocol = httpx.URL(target).scheme
    if protocol not in {"rtsp", "rtmp"}:
        raise ValueError("Unsupported output protocol")
    buffer = StreamBuffer(config["frame_buffer_size"], config["audio_buffer_seconds"], config["audio_buffer_bytes"], stop)
    writer_done = threading.Event()
    metrics = StreamMetrics()
    redactor = FrameRedactor(detector, mode=config.get("redaction_mode", "solid"),
                             detection_interval=config.get("detection_interval", 1), tracking=config.get("tracking", "template"))
    frames = None
    cuda_device = None
    if config.get("inference_device") == "cuda":
        import torch

        cuda_device = torch.cuda.current_device()

    def read_source():
        nonlocal frames
        source_reader = None
        try:
            hardware = {}
            if cuda_device is not None:
                from vision_shield.media.cuda_frames import CUDAFrames, hwaccel

                # Establish the current CUDA context in the thread that opens NVDEC.
                torch.cuda.set_device(cuda_device)
                frames = CUDAFrames()
                hardware["hwaccel"] = hwaccel(frames.device)
            # RTSP SDP supplies codec parameters; long probing creates an audio burst.
            # shortcut: video-only RTCP clock trial; restore audio after timing recovery is validated.
            source_reader = av.open(source, options={"rtsp_transport": "tcp", "analyzeduration": "100000", "probesize": "32768",
                                                    "allowed_media_types": "video"},
                                    timeout=(config["rtsp_connect_timeout"], config["rtsp_read_timeout"]), **hardware)
            video = check_streams(source_reader, config, audio_codecs=("aac", "pcm_alaw", "pcm_mulaw"))
            with buffer.condition:
                buffer.input_video = video
                buffer.input_audio = source_reader.streams.audio[0] if source_reader.streams.audio else None
            received = {}
            for packet in source_reader.demux():
                if stop.is_set():
                    break
                if packet.stream.type == "video":
                    if packet.pts is not None and packet.time_base is not None:
                        received.setdefault(packet.pts * packet.time_base, time.monotonic())
                        if len(received) > 256:
                            del received[next(iter(received))]
                    for frame in packet.decode():
                        check_dimensions(frame.width, frame.height, config)
                        moment = frame.pts * frame.time_base if frame.pts is not None and frame.time_base is not None else None
                        buffer.frame(frame, received.pop(moment, None))
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
    video, last_stats, failed_frames, processing = None, None, 0, False

    def report(force=False):
        nonlocal last_stats
        now = time.monotonic()
        if not force and last_stats is not None and now - last_stats < 1:
            return
        with buffer.condition:
            dropped = buffer.dropped
            stream_stats = dict(input_frames=buffer.input_frames, queue_frames=len(buffer.frames),
                               audio_queue_bytes=buffer.audio_size, failed_frames=failed_frames,
                               protected_frames=0, **metrics.snapshot())
        emit({**({"video_decoder": "h264_nvdec" if frames else "h264",
                  "video_encoder": video.codec_context.name} if video else {}),
              "detection_stats": redactor.detection_stats.copy(),
              "stats": {"processed_frames": count, "dropped_frames": dropped,
                        "elapsed_seconds": now - started, "stream": stream_stats},
              "timings": dict(frame=frame_seconds, inference=redactor.inference_seconds,
                              encode=encode_seconds, **redactor.candidate_timings, **snapshot(detector))})
        last_stats = now

    def mux_video(packet):
        nonlocal failed_frames, processing
        # Capture the encoder clock before mux rescales it (notably FLV milliseconds).
        moment = packet.pts * packet.time_base if packet.pts is not None and packet.time_base is not None else None
        try:
            writer.mux(packet)
        except Exception:
            failed_frames += 1
            processing = False
            raise
        metrics.written(moment, time.monotonic())

    try:
        options = {"rtsp_transport": "tcp"}
        if protocol == "rtmp":
            original = httpx.URL(config.get("publish_url") or target)
            # DNS 固定到授权 IP，RTMP tcUrl 仍保留平台用于虚拟主机路由的原域名。
            options = {"rtmp_tcurl": str(original.copy_with(path=original.path.rsplit("/", 1)[0],
                       query=None, username="", password="")),
                       "rw_timeout": str(round(config["rtsp_read_timeout"] * 1_000_000))}
        with av.open(target, "w", format="flv" if protocol == "rtmp" else "rtsp", options=options,
                     timeout=(config["rtsp_connect_timeout"], config["rtsp_read_timeout"])) as writer:
            video, audio = None, None
            transcode_audio = False
            while sample := buffer.take():
                frame, packets = sample
                processing = True
                moment = frame.pts * frame.time_base
                if previous is not None and moment <= previous:
                    raise BufferError("Stream clock moved backwards")
                previous = moment
                if origin is None:
                    origin = moment
                stage = time.monotonic()
                if frames:
                    output = frames.masked(frame, redactor)
                    output.pts = round((moment - origin) / output.time_base)
                else:
                    output = masked_frame(frame, redactor, origin)
                frame_seconds += time.monotonic() - stage
                stage = time.monotonic()
                if video is None:
                    video = add_video(writer, output, buffer.input_video, live=True, device=config.get("inference_device", "cpu"))
                    if buffer.input_audio:
                        transcode_audio = protocol == "rtmp" and buffer.input_audio.codec_context.name != "aac"
                        if transcode_audio:
                            audio = writer.add_stream("aac", rate=48000)
                            audio.layout = "mono" if buffer.input_audio.codec_context.channels == 1 else "stereo"
                        else:
                            audio = writer.add_stream_from_template(buffer.input_audio)
                if (output.width, output.height) != (video.width, video.height):
                    raise BufferError("Stream dimensions changed")
                for packet in packets:
                    packet.pts -= round(origin / packet.time_base)
                    packet.dts -= round(origin / packet.time_base)
                    if transcode_audio:
                        for frame in packet.decode():
                            for encoded in audio.encode(frame):
                                writer.mux(encoded)
                    else:
                        packet.stream = audio
                        writer.mux(packet)
                published = False
                metrics.submit(output, buffer.last_received, moment)
                for packet in video.encode(output):
                    mux_video(packet)
                    published = True
                encode_seconds += time.monotonic() - stage
                # 只有视频包实际写入成功才证明输出；合并到至多每五秒一次。
                if published and (last_output_report is None or time.monotonic() - last_output_report >= 5):
                    emit({"output": True, "output_monotonic": time.monotonic(), "ready": True})
                    last_output_report = time.monotonic()
                count += 1
                processing = False
                report()
                if stop.is_set():
                    break
            if video:
                for packet in video.encode():
                    mux_video(packet)
            if transcode_audio:
                for packet in audio.encode():
                    writer.mux(packet)
    except Exception:
        failed_frames += int(processing)
        raise
    finally:
        stop.set()
        writer_done.set()
        reader.join(timeout=config["rtsp_read_timeout"] + 1)
        report(force=True)
