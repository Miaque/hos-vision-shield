"""One RTSP input, bounded FIFO backpressure, and one output mux owner."""

from collections import deque
from fractions import Fraction
from math import floor
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import av
import httpx

from vision_shield.media.video import add_video, check_streams, check_dimensions, masked_frame
from vision_shield.media.stream_metrics import StreamMetrics, vcl_digest
from vision_shield.media.attempts import failure_details
from vision_shield.redaction import FrameRedactor
from vision_shield.detection.timing import measure, snapshot
if TYPE_CHECKING:
    from vision_shield.detection import Detector


class StreamCleanupError(OSError):
    """A timed-out input reader requires process reclamation before reuse."""


class LiveAVClock:
    """在有序读包线程修复两轨时钟，避免原始回退时间阻塞音视频配对。"""

    def __init__(self, video=None, audio=None):
        self.audio_time = self.audio_source = self.audio_duration = self.audio_step = None
        self.video_time = self.video_source = self.video_step = None
        self.audio_regressed = self.video_regressed = False
        self.audio_offset = 0
        self.audio_codec = audio.codec_context if audio else None
        rate = getattr(video, "average_rate", None)
        self.video_interval = 1 / rate if rate and rate > 0 else None

    def audio_packet(self, packet):
        """按可靠包间隔恢复音频时钟，保持压缩内容及 PTS/DTS 相对关系。"""
        if packet.pts is None or packet.dts is None or packet.time_base is None:
            raise BufferError("Invalid audio clock")
        source = packet.dts * packet.time_base
        duration = (packet.duration or 0) * packet.time_base
        rate = getattr(self.audio_codec, "sample_rate", 0)
        samples = getattr(self.audio_codec, "frame_size", 0)
        channels = getattr(self.audio_codec, "channels", 0)
        # RTMP 时基只有毫秒；从真实采样数保留小数，避免每包取整逐渐累积音画偏移。
        if rate and samples:
            duration = Fraction(samples, rate)
        elif rate and channels and self.audio_codec.name in {"pcm_alaw", "pcm_mulaw"}:
            duration = Fraction(packet.size, rate * channels)
        if self.audio_source is not None and source > self.audio_source and not self.audio_regressed:
            self.audio_step = source - self.audio_source
        self.audio_regressed = self.audio_source is not None and source <= self.audio_source
        moment = source
        if self.audio_time is not None and source <= self.audio_time:
            step = self.audio_duration or duration or self.audio_step
            if step is None or step <= 0:
                raise BufferError("Cannot repair audio clock without a packet interval")
            moment = self.audio_time + step
            corrected = max(round(moment / packet.time_base),
                            floor(self.audio_time / packet.time_base) + 1)
            packet.pts += corrected - packet.dts
            packet.dts = corrected
        self.audio_time = moment
        self.audio_offset = self.audio_time - source
        self.audio_source = source
        self.audio_duration = duration or self.audio_step

    def video_frame(self, frame):
        """视频回退时参考音频修复偏移，孤立坏 PTS 恢复后不积累音画偏移。"""
        source = frame.pts * frame.time_base
        if self.video_source is not None and source > self.video_source and not self.video_regressed:
            self.video_step = source - self.video_source
        self.video_regressed = self.video_source is not None and source <= self.video_source
        self.video_source = source
        moment = source
        if self.video_time is not None and source <= self.video_time:
            moment = source + self.audio_offset
            if moment <= self.video_time:
                duration = getattr(frame, "duration", 0) * frame.time_base
                if self.video_interval and (not duration or abs(duration - self.video_interval) < frame.time_base):
                    duration = self.video_interval
                step = duration or self.video_step
                if step is None or step <= 0:
                    raise BufferError("Cannot repair clock without a frame interval")
                moment = self.video_time + step
            frame.pts = max(round(moment / frame.time_base), floor(self.video_time / frame.time_base) + 1)
        self.video_time = moment
        return frame.pts * frame.time_base != source


class StreamBuffer:
    def __init__(self, frame_limit, audio_seconds, audio_bytes, stop):
        self.frame_limit, self.audio_seconds, self.audio_limit = frame_limit, audio_seconds, audio_bytes
        self.stop, self.condition = stop, threading.Condition()
        self.frames, self.audio = deque(), deque()
        self.received, self.last_received, self.input_frames = deque(), None, 0
        self.last_decoded = None
        self.last_source_pts, self.last_timestamp_repaired = None, False
        self.audio_size, self.dropped = 0, 0
        self.eof, self.error, self.input_video, self.input_audio = False, None, None, None
        self.failure = None

    def frame(self, value, received_at=None, *, source_pts=None, timestamp_repaired=False, decoded_at=None):
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
            self.received.append((received_at, source_pts, timestamp_repaired, decoded_at))
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
            self.last_received, self.last_source_pts, self.last_timestamp_repaired, self.last_decoded = self.received.popleft()
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
    """Read RTSP/RTMP and publish RTSP/RTMP; emit only processed frames and bounded stats."""
    protocol = httpx.URL(target).scheme
    if protocol not in {"rtsp", "rtmp"}:
        raise ValueError("Unsupported output protocol")
    buffer = StreamBuffer(config["frame_buffer_size"], config["audio_buffer_seconds"], config["audio_buffer_bytes"], stop)
    writer_done = threading.Event()
    debug = config.get('debug', False)
    audio_enabled = config.get("audio_enabled", True)
    metrics = StreamMetrics(started=config.get('stream_started_monotonic'), debug=debug)
    interval = config.get('detection_interval', 3)
    redactor = FrameRedactor(detector, mode=config.get("redaction_mode", "solid"),
                               detection_interval=interval,
                               tracking=config.get('tracking', 'flow' if interval == 3 else 'template'), debug=debug)
    frames = None
    cuda_device = None
    if config.get("inference_device") == "cuda":
        import torch

        cuda_device = torch.cuda.current_device()

    def read_source():
        nonlocal frames
        source_reader = None
        phase = "source_initialize"
        try:
            hardware = {}
            if cuda_device is not None:
                from vision_shield.media.cuda_frames import CUDAFrames, hwaccel

                # Establish the current CUDA context in the thread that opens NVDEC.
                torch.cuda.set_device(cuda_device)
                frames = CUDAFrames()
                hardware["hwaccel"] = hwaccel(frames.device)
            # RTSP SDP 已提供编码参数；按声音开关协商音轨，避免长时间探测造成音频突发。
            phase = "source_open"
            options = {"rtsp_transport": "tcp", "analyzeduration": "100000", "probesize": "32768",
                       "allowed_media_types": "video+audio" if audio_enabled else "video"}
            if httpx.URL(source).scheme == "rtmp":
                original = httpx.URL(config.get("source_url") or source)
                options = {"rtmp_tcurl": str(original.copy_with(path=original.path.rsplit("/", 1)[0],
                           query=None, username="", password="")),
                           "rw_timeout": str(round(config["rtsp_read_timeout"] * 1_000_000))}
            source_reader = av.open(source, options=options,
                                    timeout=(config["rtsp_connect_timeout"], config["rtsp_read_timeout"]), **hardware)
            phase = "source_validate"
            video = check_streams(source_reader, config, audio_codecs=("aac", "pcm_alaw", "pcm_mulaw"),
                                  audio_enabled=audio_enabled)
            with buffer.condition:
                buffer.input_video = video
                buffer.input_audio = source_reader.streams.audio[0] if audio_enabled and source_reader.streams.audio else None
            av_clock = LiveAVClock(video, buffer.input_audio) if buffer.input_audio else None
            received = {}
            # 静音任务只读取视频，不让音频时钟和缓冲阻止视频时间戳修复。
            packets = iter(source_reader.demux() if audio_enabled else source_reader.demux(video=0))
            while True:
                try:
                    phase = "source_read"
                    with measure(redactor, "source-read"):
                        packet = next(packets)
                except StopIteration:
                    break
                if stop.is_set():
                    break
                if packet.stream.type == "video":
                    if debug and packet.pts is not None and packet.time_base is not None:
                        received.setdefault(packet.pts * packet.time_base, time.monotonic())
                        if len(received) > 256:
                            del received[next(iter(received))]
                    phase = "source_decode"
                    with measure(redactor, "source-decode"):
                        decoded = packet.decode()
                    decoded_at = time.monotonic() if debug else None
                    for frame in decoded:
                        check_dimensions(frame.width, frame.height, config)
                        moment = frame.pts * frame.time_base if frame.pts is not None and frame.time_base is not None else None
                        received_at = received.pop(moment, None)
                        phase = "input_clock"
                        repaired = av_clock.video_frame(frame) if av_clock and moment is not None else False
                        phase = "input_buffer"
                        with measure(redactor, "source-backpressure"):
                            buffer.frame(frame, received_at, source_pts=moment, timestamp_repaired=repaired, decoded_at=decoded_at)
                elif audio_enabled and packet.stream.type == "audio" and packet.dts is not None:
                    phase = "input_audio_clock"
                    if av_clock:
                        av_clock.audio_packet(packet)
                    phase = "input_audio_buffer"
                    buffer.audio_packet(packet)
        except Exception as error:
            with buffer.condition:
                buffer.error = error
                buffer.failure = failure_details(error, phase)
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
    previous_source, step, clock_offset = None, None, 0
    started = time.monotonic()
    last_output_report = None
    frame_seconds = encode_seconds = 0.0
    video, last_stats, failed_frames, processing = None, None, 0, False
    phase, ending = "publish_open", None

    def report(force=False):
        nonlocal last_stats
        now = time.monotonic()
        if not force and last_stats is not None and now - last_stats < 1:
            return
        with buffer.condition:
            dropped = buffer.dropped
            stream_stats = dict(input_frames=buffer.input_frames, queue_frames=len(buffer.frames),
                               **(dict(input_width=buffer.input_video.width, input_height=buffer.input_video.height)
                                  if buffer.input_video else {}),
                               audio_queue_bytes=buffer.audio_size, failed_frames=failed_frames,
                               protected_frames=0, **metrics.snapshot()) if debug else None
        emit({**({"stream_end": ending} if force and ending else {}),
              **({"video_decoder": "h264_nvdec" if frames else "h264",
                  "video_encoder": video.codec_context.name} if video else {}),
              "stats": {"processed_frames": count, "dropped_frames": dropped,
                        "elapsed_seconds": now - started, **({"stream": stream_stats} if debug else {})},
              **({"detection_stats": redactor.detection_stats.copy(),
              "timings": dict(frame=frame_seconds, inference=redactor.inference_seconds,
                              encode=encode_seconds, **redactor.candidate_timings, **snapshot(detector))} if debug else {})})
        last_stats = now

    def mux_video(packet):
        nonlocal failed_frames, processing, ending
        # Capture the encoder clock before mux rescales it (notably FLV milliseconds).
        moment = packet.pts * packet.time_base if packet.pts is not None and packet.time_base is not None else None
        encoded_bytes = packet.size if debug else None
        encoded_vcl_sha256 = vcl_digest(bytes(packet)) if debug else None
        try:
            writer.mux(packet)
        except Exception as error:
            failed_frames += 1
            processing = False
            ending = failure_details(error, "publish_write")
            raise
        metrics.written(moment, time.monotonic(), encoded_bytes=encoded_bytes, encoded_vcl_sha256=encoded_vcl_sha256)

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
                processing_at = time.monotonic() if debug else None
                frame, packets = sample
                processing = True
                phase = "input_clock"
                input_moment = frame.pts * frame.time_base
                source_moment = buffer.last_source_pts if buffer.last_source_pts is not None else input_moment
                if previous_source is not None and source_moment > previous_source:
                    step = source_moment - previous_source
                previous_source = source_moment
                frame.pts = round((input_moment + clock_offset) / frame.time_base)
                moment = frame.pts * frame.time_base
                regressed = previous is not None and moment <= previous
                repaired = buffer.last_timestamp_repaired or regressed
                if regressed:
                    if buffer.input_audio:
                        raise BufferError("Invalid normalized audiovisual clock")
                    duration = getattr(frame, "duration", 0) * frame.time_base or step
                    if duration is None or duration <= 0:
                        raise BufferError("Cannot repair clock without a frame interval")
                    # Keep independent pictures; retain the resulting offset until the next discontinuity.
                    frame.pts = floor(previous / frame.time_base) + max(1, round(duration / frame.time_base))
                    moment = frame.pts * frame.time_base
                    clock_offset = moment - source_moment
                previous = moment
                if origin is None:
                    origin = moment
                stage = time.monotonic() if debug else None
                phase = "frame_process"
                if frames:
                    output = frames.masked(frame, redactor)
                    output.pts = round((moment - origin) / output.time_base)
                else:
                    output = masked_frame(frame, redactor, origin)
                if debug:
                    frame_seconds += time.monotonic() - stage
                    stage = time.monotonic()
                if video is None:
                    phase = "encoder_initialize"
                    video_options = dict(live=True, device=config.get("inference_device", "cpu"))
                    if config.get("max_bitrate") is not None:
                        video_options["max_bitrate"] = config["max_bitrate"]
                    video = add_video(writer, output, buffer.input_video, **video_options)
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
                    phase = "audio_process"
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
                phase = "video_encode"
                metrics.submit(output, buffer.last_received, source_moment, timeline_pts=moment, timestamp_repaired=repaired,
                               decoded_at=buffer.last_decoded, processing_at=processing_at)
                for packet in video.encode(output):
                    mux_video(packet)
                    published = True
                if debug:
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
                phase = "input_wait"
            if video:
                phase = "encoder_flush"
                for packet in video.encode():
                    mux_video(packet)
            if transcode_audio:
                phase = "audio_flush"
                for packet in audio.encode():
                    writer.mux(packet)
            phase = "publish_close"
    except Exception as error:
        failed_frames += int(processing)
        if ending is None:
            ending = buffer.failure if error is buffer.error else failure_details(error, phase)
        raise
    finally:
        if ending is None:
            ending = dict(reason="stop_requested" if stop.is_set() else "source_eof", phase="source_read")
        ending.update(attempt_id=metrics.attempt_id, ended_monotonic=time.monotonic())
        stop.set()
        writer_done.set()
        reader.join(timeout=config["rtsp_read_timeout"] + 1)
        report(force=True)
        if reader.is_alive():
            raise StreamCleanupError("Source reader did not stop")
