"""Same decoded pictures, masks and forced incomplete tails; no FPS acceptance claim."""
import hashlib
import json
from pathlib import Path
import sys
import os

import av
import torch

from vision_shield.config import Settings
from vision_shield.detection import load_detector
from vision_shield.media.cuda_frames import CUDAFrames, hwaccel
from vision_shield.media.plate_lookahead import PlateLookahead
from vision_shield.redaction import FrameRedactor


def compare(source, output):
    cfg = Settings().model_dump(mode='json')
    cfg['debug'] = False
    detectors = [load_detector(cfg) for _ in range(2)]
    detectors[0].primary.models[1]._graphed = None
    if os.getenv('QUALITY_ALL_ORIGINAL') == '1':
        detectors[1].primary.models[1]._graphed = None
    redactors = [FrameRedactor(d, mode='mosaic', detection_interval=3, tracking='flow') for d in detectors]
    delays = [PlateLookahead(r) for r in redactors]
    processors = [CUDAFrames() for _ in range(2)]
    results, counts = [[], []], [0, 0]

    def retain(index, ready):
        for frame in ready:
            planes = [torch.from_dlpack(p.__dlpack__()).cpu().numpy().tobytes() for p in frame.planes]
            results[index].append((str(frame.pts * frame.time_base), hashlib.sha256(b''.join(planes)).hexdigest()))

    av.logging.set_level(av.logging.ERROR)
    with av.open(source, hwaccel=hwaccel(processors[0].device)) as reader, av.logging.Capture() as logs:
        decoded, number = iter(reader.decode(video=0)), 0
        while True:
            try:
                frame = next(decoded)
            except StopIteration:
                break
            except Exception as error:
                print(json.dumps(dict(frame=number, error=type(error).__name__, logs=logs[-8:])), flush=True)
                raise
            number += 1
            if frame.pts is None:
                continue
            for i in range(2):
                pending = processors[i].masked(frame, redactors[i], defer=True)
                # A deterministic incomplete-window schedule compares safety paths on identical pictures.
                pending.deadline = None
                retain(i, delays[i].push(pending))
                if number % 97 == 0:
                    retain(i, delays[i].flush(reason='quality-incomplete-window'))
                counts[i] += 1
            if number % 200 == 0:
                print(json.dumps(dict(processed=number, retained=[len(d.pending) for d in delays])), flush=True)
    for i in range(2):
        retain(i, delays[i].flush(reason='eof'))
    assert counts[0] == counts[1] == len(results[0]) == len(results[1])
    mismatches = [i + 1 for i, (a, b) in enumerate(zip(*results, strict=True)) if a != b]
    report = dict(source=Path(source).name, frames=counts[0], identical_yuv_and_pts=not mismatches,
                  mismatches=mismatches, forced_incomplete_flushes=counts[0] // 97,
                  output_digest=hashlib.sha256(json.dumps(results[1]).encode()).hexdigest(),
                  note='Quality equivalence on source pictures; not a live throughput measurement')
    assert not mismatches, report
    print(json.dumps(report), flush=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    for detector in detectors:
        for owner in (detector, detector.detail):
            if pool := vars(owner).get('_pool'):
                pool.shutdown()


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
