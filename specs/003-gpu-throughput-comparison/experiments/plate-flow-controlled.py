"""Controlled P0/P2-A replay; host diagnostics are not performance acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from contextlib import ExitStack
from unittest.mock import patch

assert len(sys.argv) == 8, 'TREE INPUT OUTPUT host|normal baseline|candidate SOURCE_START MODELS'

tree, source, directory = map(Path, sys.argv[1:4])
observed = sys.argv[4] == 'host'
assert sys.argv[4] in ('host', 'normal')
source_start = int(sys.argv[6])
assert source_start >= 1
candidate = sys.argv[5] == 'candidate'
assert sys.argv[5] in ('baseline', 'candidate')
directory.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(tree))
import av
real_open = av.open
import cv2
import torch
from vision_shield.config import Settings
from vision_shield.detection import load_detector
from vision_shield.detection.egoblur import _HalfPlate
from vision_shield.detection.host_profile import CONTEXT, ProfileTimes, instrument, instrument_candidates, runtime_metadata
from vision_shield.media.video import process_video
from vision_shield.media import streams
from vision_shield.redaction import FrameRedactor

assets = Path(sys.argv[7]).resolve()
config = Settings(_env_file=None, inference_device='cuda', detection_backend='quality_yolo', debug=observed,
    face_model_path=assets/'yunet_face.onnx', plate_model_path=assets/'hyperlpr_plate.onnx',
    egoblur_plate_model_path=assets/'license_plate.jit').model_dump()
config.update(detection_interval=3, tracking='flow', redaction_mode='solid', plate_flow_recovery=candidate)
torch.set_num_threads(4)
cv2.setNumThreads(2)
torch.cuda.set_device(0)
last = {}
next_progress = 0.0

def emit(event):
    global next_progress
    last.update(event)
    now = time.perf_counter()
    if now >= next_progress and 'stats' in event:
        print(json.dumps(dict(progress=event['stats'])), flush=True)
        next_progress = now + 10

def media(path):
    with real_open(str(path)) as reader:
        frames = [(f.width, f.height, str(f.pts*f.time_base)) for f in reader.decode(video=0)]
    with real_open(str(path)) as reader:
        audio = [(str(p.pts*p.time_base), str(p.dts*p.time_base), str(p.duration*p.time_base),
                  hashlib.sha256(bytes(p)).hexdigest()) for p in reader.demux(audio=0) if p.dts is not None] if reader.streams.audio else []
    return frames, audio

with ExitStack() as stack:
    collector = None
    if observed:
        masks = stack.enter_context((directory/'masks.jsonl').open('x', encoding='utf-8'))
        tracks = stack.enter_context((directory/'tracks.jsonl').open('x', encoding='utf-8'))
        failures = stack.enter_context((directory/'flow-failures.jsonl').open('x', encoding='utf-8'))
        decisions = stack.enter_context((directory/'decisions.jsonl').open('x', encoding='utf-8'))
        whole_events = stack.enter_context((directory/'whole-events.jsonl').open('x', encoding='utf-8'))
        raw_events = stack.enter_context((directory/'raw-flow-events.jsonl').open('x', encoding='utf-8'))

        class Evidence(ProfileTimes):
            def record_masks(self, shape, boxes):
                masks.write(json.dumps(dict(frame=CONTEXT.get()['frame'], boxes=boxes), default=float)+'\n')
                super().record_masks(shape, boxes)

        collector = Evidence(directory/'host-profile')
        stack.callback(collector.close)
        instrument(stack, collector)
        original = FrameRedactor.regions
        flow = FrameRedactor._flow
        detect = FrameRedactor._detections

        def detection_decision(self, *args, **kwargs):
            caller = sys._getframe(1)
            if caller.f_code.co_name == 'regions':
                state = caller.f_locals
                decisions.write(json.dumps(dict(frame=self.frame_index, reset=state['reset'],
                    periodic=(self.frame_index-1)%self.detection_interval == 0,
                    recheck=state['recheck'], face_only=state['face_only'],
                    failed_tracks=sum(b is None for b in state['flow_boxes'].values())), default=float)+'\n')
            return detect(self, *args, **kwargs)

        def regions(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            tracks.write(json.dumps(dict(frame=self.frame_index, moment=self.moment,
                tracks=[dict(kind=t.kind, box=t.box, seen=t.seen) for t in self.history],
                pending=getattr(self.detector, 'unconfirmed_plate_boxes', None),
                stats=self.detection_stats), default=float)+'\n')
            return result

        def checked_flow(self, gray, box, reference=None, **kwargs):
            before = time.perf_counter()
            result = flow(self, gray, box, reference, **kwargs)
            if kwargs.get("whole_box"):
                whole_events.write(json.dumps(dict(frame=self.frame_index, moment=self.moment, box=box, result=result, host_ms=(time.perf_counter()-before)*1000),default=float)+"\n")
            track = next((t for t in self.history if t.box == box), None)
            if not kwargs.get('whole_box'):
                raw_events.write(json.dumps(dict(frame=self.frame_index, moment=self.moment, box=box, result=result,
                    kind=track.kind if track else None, seen=track.seen if track else None),default=float)+'\n')
            if not kwargs.get("whole_box") and result is None and track and self._plate_template_fallback and track.kind == 'plate' and self.moment-track.seen <= .2:
                failures.write(json.dumps(dict(frame=self.frame_index, moment=self.moment, box=box,
                    age=self.moment-track.seen, global_blocked=bool(getattr(self.detector, 'has_unconfirmed_plates', False))), default=float)+'\n')
            return result

        stack.enter_context(patch.object(FrameRedactor, 'regions', regions))
        stack.enter_context(patch.object(FrameRedactor, '_flow', checked_flow))
        stack.enter_context(patch.object(FrameRedactor, '_detections', detection_decision))
    original_init = FrameRedactor.__init__
    def aligned_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.frame_index = source_start-1
    stack.enter_context(patch.object(FrameRedactor, '__init__', aligned_init))
    started = time.monotonic()
    detector = load_detector(config)
    if not isinstance(detector.primary.models[1], _HalfPlate):
        detector.primary.models[1] = _HalfPlate(detector.primary.models[1])
    loading = time.monotonic()-started
    if observed:
        instrument_candidates(detector.primary, 'primary', collector)
        instrument_candidates(detector.detail, 'detail', collector)
    metadata = runtime_metadata(detector, config)
    metadata['canonical_source_sha256'] = {name: hashlib.sha256(Path(sys.modules['vision_shield.'+name].__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in metadata['source_sha256']}
    (directory/'manifest.json').write_text(json.dumps(dict(source=str(source), tree=str(tree), observed=observed, candidate=candidate,
        pid=os.getpid(), mode=sys.argv[4],
        scope='Controlled local GTX 970 with original P0 FP16 wrapper and identical weights. Candidate adopts gated last-resort whole LK. Host diagnostics and clip timings are not formal FPS evidence.',
        runtime=metadata, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2, default=str), 'utf-8')
    torch.cuda.synchronize()
    started, cpu_started = time.perf_counter(), time.process_time()
    (directory/'processing-start.json').write_text(json.dumps(dict(perf_counter=started, cpu_seconds=cpu_started)), 'utf-8')
    count = process_video(source, directory/'redacted.mp4', detector, config, threading.Event(), emit)
    torch.cuda.synchronize()
    cpu_seconds, elapsed = time.process_time()-cpu_started, time.perf_counter()-started
    (directory/'processing-end.json').write_text(json.dumps(dict(perf_counter=started+elapsed)), 'utf-8')
    expected = media(source)
    actual = media(directory/'redacted.mp4')
    assert actual == expected, 'frame/PTS/audio integrity changed'
    assert count == len(expected[0])
    if observed:
        masks.flush()
        assert len((directory/'masks.jsonl').read_text('utf-8').splitlines()) == count
    outcome = dict(frames=count, audio_packets=len(actual[1]), media_seconds=elapsed, fps=count/elapsed,
        cpu_process_seconds=cpu_seconds, cpu_core_percent=cpu_seconds/elapsed*100,
        load_seconds=loading, integrity='all frame dimensions/PTS and AAC payloads/clocks equal input', last=last,
        profile=collector.report(count) if collector else None)
    (directory/'outcome.json').write_text(json.dumps(outcome, indent=2, default=str), 'utf-8')
    print(json.dumps({k:v for k,v in outcome.items() if k not in ('profile', 'last')}), flush=True)
