"""Incremental E0 host overhead, excluding models, GPU and media I/O."""
from contextlib import nullcontext
from fractions import Fraction
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from types import SimpleNamespace

from vision_shield.detection.timing import StageTimes
from vision_shield.media.stream_e0 import StreamE0
from vision_shield.media.stream_pacing import StreamPacer
from vision_shield.media.streams import StreamBuffer
from vision_shield.media.live_diagnostics import LiveDiagnostics


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def is_set(self):
        return False
    def wait(self, seconds):
        self.now += seconds
        return False


def run(enabled, count=2000):
    clock = Clock()
    trace = StreamE0(clock=clock) if enabled else None
    stages = StageTimes()
    stages.span_observer = trace.span if trace else None
    buffer = StreamBuffer(2, 16, 4194304, clock)
    buffer.trace = trace
    pacer = StreamPacer(clock, clock=clock)
    pacer.observer = trace.pacing if trace else None
    wall, cpu = time.perf_counter(), time.process_time()
    for i in range(count):
        frame = SimpleNamespace(pts=3600 * i, dts=3600 * i, time_base=Fraction(1, 90000), duration=3600)
        fid = trace.frame_start(frame, clock.now, decoded_at=clock.now,
                                packet_clock=trace.media_clock(frame)) if trace else None
        if trace:
            trace.video_packet(trace.media_clock(frame))
            trace.video_clock(fid, 'input_clock', frame)
        buffer.frame(frame, trace_id=fid)
        buffer.take()
        if trace:
            trace.video_clock(fid, 'normalized_clock', frame)
        with trace.span('op:process') if trace else nullcontext():
            for _ in range(12):
                with stages.measure('track-recheck'):
                    with stages.measure('forward-primary-plate'):
                        pass
        if trace:
            for name in ('lookahead_entered', 'lookahead_released', 'render_started', 'render_completed',
                         'encode_started', 'encode_completed', 'mux_started'):
                trace.update_frame(fid, **{name + '_monotonic': clock.now})
            trace.encode_context(frame.pts * frame.time_base, fid, encoder_frame_clock=trace.media_clock(frame), origin='0')
            trace.packet_clock = trace.media_clock(frame)
        with trace.span('op:output-pacing') if trace else nullcontext():
            assert pacer.wait(frame.pts * frame.time_base)
        if trace:
            trace.written(frame.pts * frame.time_base, clock.now)
            if i % 3 == 0:
                trace.audio_clock(trace.media_clock(frame), trace.media_clock(frame))
                row = trace.audio_publish(trace.media_clock(frame), trace.media_clock(frame), 0, False)
                trace.audio_accepted(row)
    return trace, dict(wall_ms_per_frame=(time.perf_counter()-wall)*1000/count,
                       cpu_ms_per_frame=(time.process_time()-cpu)*1000/count)


def main():
    rounds = []
    for _ in range(3):
        for enabled in (False, True):
            gc.collect()
            trace, timing = run(enabled)
            rounds.append(dict(enabled=enabled, **timing))
    snapshots = []
    for _ in range(10):
        before = time.perf_counter()
        state = trace.snapshot()
        snapshots.append((time.perf_counter()-before)*1000)
    legacy = LiveDiagnostics(SimpleNamespace(), {'stream_e0_diagnostics': False}, 'benchmark', runtime={'metadata': 'x'*12000})
    legacy.e0 = trace
    for i in range(128):
        legacy.operations.append(dict(sequence=i, kind='process', seconds=.04,
            stages={f'stage{j}': .001 for j in range(30)}))
    legacy.operation_count = 128
    combined = []
    for _ in range(5):
        before = time.perf_counter()
        wire = legacy.snapshot()
        combined.append((time.perf_counter()-before)*1000)
    assert len(json.dumps(wire).encode()) < 192*1024
    tracemalloc.start()
    retained, _ = run(True)
    retained.snapshot()
    memory, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    before = statistics.median(r['cpu_ms_per_frame'] for r in rounds if not r['enabled'])
    after = statistics.median(r['cpu_ms_per_frame'] for r in rounds if r['enabled'])
    result = dict(rounds=rounds, incremental_cpu_ms_per_frame=after-before,
        snapshot_ms=dict(median=statistics.median(snapshots), max=max(snapshots)),
        combined_snapshot_ms=dict(median=statistics.median(combined), max=max(combined)),
        combined_snapshot_bytes=len(json.dumps(wire).encode()),
        snapshot_bytes=len(json.dumps(state).encode()), traced_retained_bytes=memory, traced_peak_bytes=peak,
        scope_calls_per_frame=24, frames_per_round=2000,
        limitations='Synthetic host probes; does not measure RTX/GPU contention, Redis or subprocess transport')
    assert state['pacer_totals']['decisions'] == 2000
    assert result['snapshot_bytes'] <= 128*1024 and peak < 8*1024*1024
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
