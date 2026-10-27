"""Bounded, uniquely clocked output samples; consumers detect gaps by sequence."""

from collections import deque
import hashlib
import re
import time
from uuid import uuid4


def vcl_digest(payload):
    """Hash H.264 picture NALs across Annex B prefix and RTP parameter-set changes."""
    units = [unit.rstrip(b'\x00') for unit in re.split(b'\x00\x00\x00?\x01', payload)[1:]
             if unit and unit[0] & 31 in (1, 5)]
    if not units:
        return None
    digest = hashlib.sha256()
    for unit in units:
        digest.update(len(unit).to_bytes(4, 'big'))
        digest.update(unit)
    return digest.hexdigest()


class StreamMetrics:
    def __init__(self, *, started=None, debug=False):
        self.debug = debug
        self.attempt_id = uuid4().hex
        self.started = time.monotonic() if started is None else started
        self.pending, self.outputs = {}, deque(maxlen=512)
        self.frames = self.packets = self.unmatched = self.missing_receive = 0
        self.repairs = 0

    def submit(self, frame, received_at, source_pts, *, timeline_pts=None, timestamp_repaired=False,
               decoded_at=None, processing_at=None):
        if len(self.pending) >= 512:
            raise BufferError('Encoder backlog exceeded limit')
        moment = frame.pts * frame.time_base
        if moment in self.pending:
            raise BufferError('Duplicate encoder input clock')
        # Production retains encoder backlog/clock guards without collecting frame evidence.
        self.pending[moment] = ((received_at, str(source_pts), str(source_pts if timeline_pts is None else timeline_pts),
                                 timestamp_repaired, decoded_at, processing_at)
                                if self.debug else ())
        self.repairs += timestamp_repaired

    def written(self, moment, now, *, encoded_bytes=None, encoded_vcl_sha256=None):
        """Call only after mux succeeds; mux may mutate packet PTS/time_base."""
        self.packets += 1
        sample = self.pending.pop(moment, None)
        if sample is None:
            self.unmatched += 1
            return
        self.frames += 1
        if not self.debug:
            return
        received_at, source_pts, timeline_pts, repaired, decoded_at, processing_at = sample
        self.missing_receive += received_at is None
        self.outputs.append(dict(sequence=self.frames, source_pts=source_pts, timeline_pts=timeline_pts, timestamp_repaired=repaired,
            encoded_bytes=encoded_bytes, encoded_vcl_sha256=encoded_vcl_sha256,
            received_monotonic=received_at, encoded_monotonic=now,
            **({'decoded_monotonic': decoded_at} if decoded_at is not None else {}),
            **({'processing_monotonic': processing_at} if processing_at is not None else {}),
            latency_seconds=None if received_at is None else now - received_at))

    def snapshot(self):
        return dict(attempt_id=self.attempt_id, started_monotonic=self.started, sampled_monotonic=time.monotonic(),
            encoded_frames=self.frames, encoded_packets=self.packets, unmatched_packets=self.unmatched,
            missing_receive_frames=self.missing_receive, pending_encoder_frames=len(self.pending),
            timestamp_repairs=self.repairs,
            recent_outputs=list(self.outputs))
