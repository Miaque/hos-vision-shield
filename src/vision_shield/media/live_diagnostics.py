"""DEBUG-only, bounded host observations; never changes media or GPU synchronization."""
from collections import deque
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from threading import Lock, get_ident
import time
import traceback

from vision_shield.detection.timing import snapshot


class LiveDiagnostics:
    def __init__(self, detector, config, attempt_id, *, runtime=None):
        self.detector, self.attempt_id = detector, attempt_id
        self.operations, self.slow_scopes = deque(maxlen=128), deque(maxlen=64)
        self.operation_count = self.slow_count = 0
        self.frame_sequence = 0
        self.lock = Lock()
        self.timing = getattr(detector, 'stage_times', None)
        self.previous_observer = getattr(self.timing, 'observer', None)
        if runtime is None:
            runtime = getattr(detector, '_live_runtime', None)
            if runtime is None:
                try:
                    from vision_shield.detection.host_profile import runtime_metadata
                    runtime = runtime_metadata(detector, config)
                    for name in ('media.cuda_frames', 'media.stream_pacing', 'detection.timing', 'media.live_diagnostics'):
                        path = Path(__file__).parents[1].joinpath(*name.split('.')).with_suffix('.py')
                        with path.open('rb') as handle:
                            runtime['source_sha256'][name] = hashlib.file_digest(handle, 'sha256').hexdigest()
                    detector._live_runtime = runtime
                except Exception as error:
                    # Diagnostic failure is explicit and must not terminate an otherwise safe publisher.
                    runtime = {'unavailable_reason': type(error).__name__}
        self.runtime = dict(runtime, effective_config={key: config.get(key) for key in (
            'debug', 'stream_diagnostics', 'detection_interval', 'tracking', 'redaction_mode',
            'frame_buffer_size', 'audio_buffer_seconds', 'audio_buffer_bytes', 'stream_output_pacing',
            'rtsp_connect_timeout', 'rtsp_read_timeout', 'max_bitrate', 'audio_enabled')})
        if self.timing is not None:
            self.timing.observer = self.observe

    def observe(self, name, elapsed):
        if elapsed < .1:
            return
        # End is sampled after the existing scope finishes. Start is derived from its host duration.
        ended = time.monotonic()
        sample = dict(stage=name, seconds=elapsed, ended_monotonic=ended,
                      started_monotonic_estimate=ended - elapsed, thread_id=get_ident())
        if elapsed >= .5:
            sample['callers'] = [f'{Path(row.filename).name}:{row.lineno}:{row.name}'
                                 for row in traceback.extract_stack(limit=10)[:-1]]
        with self.lock:
            self.slow_scopes.append(sample)
            self.slow_count += 1

    @contextmanager
    def operation(self, kind, source_pts, *, frame_sequence=None):
        stages, started = snapshot(self.detector), time.monotonic()
        if kind == 'process':
            self.frame_sequence += 1
        try:
            yield
        finally:
            ended, after = time.monotonic(), snapshot(self.detector)
            sample = dict(sequence=self.operation_count + 1, kind=kind,
                          source_pts=None if source_pts is None else str(source_pts),
                          started_monotonic=started, ended_monotonic=ended, seconds=ended - started,
                          stages={key: value - stages.get(key, 0) for key, value in after.items()
                                  if value > stages.get(key, 0) and not key.startswith('source-')})
            if kind == 'process':
                sample['frame_sequence'] = self.frame_sequence
            elif frame_sequence is not None:
                sample['frame_sequence'] = frame_sequence
            elif kind.startswith('lookahead'):
                sample['trigger_source_pts'] = sample.pop('source_pts')
            if kind in ('process', 'render') or ended - started >= .1:
                self.operations.append(sample)
                self.operation_count += 1

    def snapshot(self):
        with self.lock:
            scopes = list(self.slow_scopes)
            slow_count = self.slow_count
        result = dict(mode='live', attempt_id=self.attempt_id, runtime=self.runtime,
                      clock='server_process_monotonic_seconds',
                      operations=list(self.operations), slow_scopes=scopes,
                      operations_discarded=self.operation_count - len(self.operations),
                      slow_scopes_discarded=slow_count - len(scopes))
        # A separate event stays below the resident process's 256 KiB JSON-line limit.
        while len(json.dumps(result).encode()) > 192 * 1024:
            key = 'slow_scopes' if len(result['slow_scopes']) > len(result['operations']) else 'operations'
            result[key].pop(0)
            result[key + '_discarded'] += 1
        return result

    def close(self):
        if self.timing is not None:
            self.timing.observer = self.previous_observer
