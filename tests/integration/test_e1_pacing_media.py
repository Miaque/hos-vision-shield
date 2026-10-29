"""Compare independent media plans against pinned live E1 with real H264/AAC/RTSP."""
from itertools import pairwise
import base64
import json
import os
from pathlib import Path
import subprocess
import re
import threading
import time

import av
import pytest
import torch

from tests.integration.test_rtsp_interleave import interleave_source
from tests.rtsp_capture import RTSPCapture
from vision_shield.config import Settings
from vision_shield.detection import Detection
from vision_shield.media import cuda_frames, streams
from vision_shield.media.plate_lookahead import PendingMask
from vision_shield.media.stream_pacing import StreamPacer
from vision_shield.redaction import redact


def baseline_pacer():
    namespace = {}
    source = subprocess.check_output(['git', 'show',
        '79ebddc22ab4d47337083f5514fd5dcd5176aa37:src/vision_shield/media/stream_pacing.py'], text=True)
    exec(source, namespace)
    return namespace['StreamPacer']


def run(source, monkeypatch, policy, scenario, costs, *, hardware=False):
    original_open, original_add_video = av.open, streams.add_video
    stop, events, clocks, published, masks, released = threading.Event(), [], [], [], [], []
    plate = (48, 16, 60, 32)
    class MeasuredPacer(policy):
        def __init__(self, stop):
            # Windows monotonic() has a coarse tick on this Python build; measure the policy
            # with the same high-resolution clock in both arms rather than rounding a 20ms floor.
            super().__init__(stop, clock=time.perf_counter)
        def wait(self, moment):
            allowed = super().wait(moment)
            if allowed:
                released.append((moment, self.previous_actual))
            return allowed
    class Reader:
        def __init__(self, **hardware_options):
            self.inner = original_open(str(source), **hardware_options)
            self.streams = self.inner.streams
        def demux(self, **kwargs):
            for packet in self.inner.demux(**kwargs):
                if scenario == 'reset' and packet.size and packet.pts*packet.time_base >= 2:
                    shift = round(1/packet.time_base)
                    packet.pts -= shift
                    packet.dts -= shift
                yield packet
        def close(self):
            self.inner.close()
    class CPUFrames:
        device = 0
        def __init__(self):
            self.index = 0
        def masked(self, frame, redactor, *, defer):
            assert defer
            index, self.index = self.index, self.index+1
            if costs:
                time.sleep(costs[index % len(costs)])
            pixels = frame.to_ndarray(format='rgb24')
            gray = pixels[:, :, 0].copy()
            def render(extra):
                boxes = [(0, 0, 32, 48), *([plate] if index >= 12 else []), *extra]
                masks.append((index, tuple(boxes)))
                result = av.VideoFrame.from_ndarray(redact(pixels, boxes, mode='solid'), format='rgb24')
                result.pts, result.time_base = frame.pts, frame.time_base
                return result
            return PendingMask(float(frame.pts*frame.time_base), gray,
                [plate] if index >= 12 else [], [plate] if index == 12 else [], render,
                reset=index == 50, confirm=lambda box: box, discover=lambda: [plate])
    class Writer:
        def __init__(self, inner):
            self.inner = inner
        def __getattr__(self, key):
            return getattr(self.inner, key)
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def mux(self, packet):
            clocks.append((packet.stream.type, packet.pts*packet.time_base, packet.dts*packet.time_base, bytes(packet)))
            self.inner.mux(packet)
            if packet.stream.type == 'video':
                published.append(time.perf_counter())
                if scenario == 'stop' and len(published) == 30:
                    stop.set()
    def open_endpoint(url, *args, **kwargs):
        return (Reader(**{k:kwargs[k] for k in ('hwaccel',) if hardware and k in kwargs})
                if url == 'rtsp://fixture/input' else Writer(original_open(url, *args, **kwargs)))
    monkeypatch.setattr(streams.av, 'open', open_endpoint)
    monkeypatch.setattr(streams, 'StreamPacer', MeasuredPacer)
    if not hardware:
        monkeypatch.setattr(cuda_frames, 'CUDAFrames', CPUFrames)
        monkeypatch.setattr(cuda_frames, 'hwaccel', lambda device: object())
        monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
        monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
        monkeypatch.setattr(streams, 'add_video', lambda writer, frame, source, **kwargs:
            original_add_video(writer, frame, source, **dict(kwargs, device='cpu')))
    class FixedGPURegions:
        def detect_regions(self, pixels, kind=None):
            assert pixels.is_cuda
            return [Detection(name, box, .99) for name, box in [('face', (0,0,32,48)), ('plate', plate)]
                    if kind is None or kind == name]
    config = Settings(_env_file=None, debug=True, stream_diagnostics=False).model_dump()
    config.update(inference_device='cuda', detection_interval=3, tracking='flow')
    with RTSPCapture() as capture:
        streams.process_stream('rtsp://fixture/input', capture.url, FixedGPURegions() if hardware else object(),
                               config, stop, events.append)
    video, audio = capture.pictures(), capture.audio()
    per_track = {kind: [(pts, dts, data) for track, pts, dts, data in clocks if track == kind]
                 for kind in ('video', 'audio')}
    assert len(video) == len(per_track['video']) == len(published)
    assert len(audio) == len(per_track['audio'])-1
    assert all(b[0] > a[0] and b[1] > a[1] for track in per_track.values() for a, b in pairwise(track))
    assert [row['payload'] for row in audio] == [row[2] for row in per_track['audio'][:len(audio)]]
    for kind, received, rate in [('video', video, 90000), ('audio', audio, 8000)]:
        offset = (received[0]['timestamp']-round(per_track[kind][0][0]*rate)) % 2**32
        assert all((row['timestamp']-round(packet[0]*rate)) % 2**32 == offset
                   for row, packet in zip(received, per_track[kind]))
    zero = [capture.clock_time(channel, received[0]['timestamp'], rate)-float(per_track[kind][0][0])
            for kind, channel, received, rate in [('video', 0, video, 90000), ('audio', 2, audio, 8000)]]
    assert abs(zero[0]-zero[1]) < .002
    decoder = av.CodecContext.create('h264', 'r')
    parameter_sets = re.search(r'sprop-parameter-sets=([^;\r\n]+)', capture.sdp)[1]
    decoder.extradata = b''.join(b'\0\0\0\1'+base64.b64decode(value) for value in parameter_sets.split(','))
    decoded = [frame for row in video for frame in decoder.decode(av.Packet(row['payload']))]
    assert len(decoded) == len(video)
    for frame in decoded:
        pixels = frame.to_ndarray(format='rgb24')
        assert pixels[:, :28].mean() < 70
        assert pixels[18:30, 50:58].mean() < 70
    if not hardware:
        assert events[-1]['detection_stats']['backfilled_plate_masks'] > 0
    assert events[-1]['stats']['dropped_frames'] == 0
    assert events[-1]['stream_end']['reason'] == ('stop_requested' if scenario=='stop' else 'source_eof')
    return dict(clocks=per_track, published=published, released=released, events=events, masks=masks,
                video=video, audio=audio)


@pytest.mark.parametrize('scenario', ['continuous', 'reset', 'stop'])
def test_independent_plan_retains_clocks_bytes_and_audio_with_actual_lookahead(interleave_source, monkeypatch, scenario):
    arms = []
    for policy in (baseline_pacer(), StreamPacer):
        with monkeypatch.context() as scoped:
            arms.append(run(interleave_source, scoped, policy, scenario, [.09, .01, .01]))
    if scenario != 'stop':
        assert len(arms[0]['video']) == len(arms[1]['video']) == 100
        assert arms[0]['clocks'] == arms[1]['clocks']
        # Different wall deadlines can change which conservative fallback rendered a frame;
        # fixed fixture regions must still produce identical final encoded pictures.
        assert [r['payload'] for r in arms[0]['video']] == [r['payload'] for r in arms[1]['video']]
    else:
        assert len(arms[0]['video']) == len(arms[1]['video']) == 30
        assert arms[0]['clocks']['video'] == arms[1]['clocks']['video']
    for arm in arms:
        # Original repaired PTS, including the reset case, stay strictly ordered.
        gaps = [b[1]-a[1] for a, b in pairwise(arm['released'])]
        assert min(gaps) >= .019


def test_independent_plan_has_repeatable_bounded_scheduling_benefit(interleave_source, monkeypatch):
    measurements = []
    for _ in range(2):
        pair = []
        for name, policy in [('deployed_e1', baseline_pacer()), ('independent_media_plan', StreamPacer)]:
            with monkeypatch.context() as scoped:
                result = run(interleave_source, scoped, policy, 'continuous', [.09, .01, .01])
            elapsed = result['published'][-1]-result['published'][0]
            pair.append(dict(policy=name, frames=len(result['published']), seconds=elapsed,
                             fps=(len(result['published'])-1)/elapsed))
        assert pair[1]['seconds'] < pair[0]['seconds']*.98
        measurements.append(pair)
    target = os.getenv('E1_MEDIA_PLAN_RESULT_PATH')
    if target:
        Path(target).write_text(json.dumps(measurements, indent=2), encoding='utf-8')


@pytest.mark.skipif(os.getenv('RUN_GPU_TESTS') != '1', reason='requires actual NVDEC/CUDA masking/NVENC')
@pytest.mark.parametrize('scenario', ['continuous', 'reset'])
def test_independent_plan_actual_gpu_keeps_video_audio_and_pts(interleave_source, monkeypatch, scenario):
    arms = []
    for policy in (baseline_pacer(), StreamPacer):
        with monkeypatch.context() as scoped:
            arms.append(run(interleave_source, scoped, policy, scenario, [], hardware=True))
    assert len(arms[0]['video']) == len(arms[1]['video']) == 100
    assert arms[0]['clocks'] == arms[1]['clocks']
    assert [r['payload'] for r in arms[0]['video']] == [r['payload'] for r in arms[1]['video']]
