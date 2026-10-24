"""Bounded, uniquely clocked output samples; consumers detect gaps by sequence."""

from collections import deque
import time
from uuid import uuid4


class StreamMetrics:
    def __init__(self):
        self.attempt_id, self.started = uuid4().hex, time.monotonic()
        self.pending, self.outputs = {}, deque(maxlen=512)
        self.frames = self.packets = self.unmatched = self.missing_receive = 0
        self.repairs = 0

    def submit(self, frame, received_at, source_pts, *, timeline_pts=None, timestamp_repaired=False):
        if len(self.pending) >= 512:
            raise BufferError('Encoder backlog exceeded limit')
        moment = frame.pts * frame.time_base
        if moment in self.pending:
            raise BufferError('Duplicate encoder input clock')
        self.pending[moment] = (received_at, str(source_pts), str(source_pts if timeline_pts is None else timeline_pts), timestamp_repaired)
        self.repairs += timestamp_repaired

    def written(self, moment, now):
        """Call only after mux succeeds; mux may mutate packet PTS/time_base."""
        self.packets += 1
        sample = self.pending.pop(moment, None)
        if sample is None:
            self.unmatched += 1
            return
        received_at, source_pts, timeline_pts, repaired = sample
        self.frames += 1
        self.missing_receive += received_at is None
        self.outputs.append(dict(sequence=self.frames, source_pts=source_pts, timeline_pts=timeline_pts, timestamp_repaired=repaired,
            received_monotonic=received_at, encoded_monotonic=now,
            latency_seconds=None if received_at is None else now - received_at))

    def snapshot(self):
        return dict(attempt_id=self.attempt_id, started_monotonic=self.started, sampled_monotonic=time.monotonic(),
            encoded_frames=self.frames, encoded_packets=self.packets, unmatched_packets=self.unmatched,
            missing_receive_frames=self.missing_receive, pending_encoder_frames=len(self.pending),
            timestamp_repairs=self.repairs,
            recent_outputs=list(self.outputs))
