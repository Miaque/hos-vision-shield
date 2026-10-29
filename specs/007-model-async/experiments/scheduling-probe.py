"""One-variable serialized/parallel model scheduling on the same decoded rfm frames."""
import argparse
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import time

import av
import torch

from vision_shield.config import Settings
from vision_shield.detection import load_detector


class InlinePool:
    def submit(self, fn, *args, **kwargs):
        result = Future()
        try:
            result.set_result(fn(*args, **kwargs))
        except BaseException as error:
            result.set_exception(error)
        return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    selected, frames = {0,250,500,750,1000,1250,1500,1750}, []
    with av.open(str(args.source)) as reader:
        for i,frame in enumerate(reader.decode(video=0)):
            if i in selected:
                frames.append(torch.as_tensor(frame.to_ndarray(format='rgb24'), device='cuda'))
    # Only the input upload is moved outside the comparison in both arms; model preparations remain.
    detector = load_detector(Settings(_env_file=None, debug=True, detection_backend='quality_yolo',
                                      inference_device='cuda').model_dump(mode='json'))
    parallel, inline = detector._pool, InlinePool()
    baseline = [repr(detector.detect_regions(frame)) for frame in frames]
    results = []
    for repeat in range(3):
        for mode in (['parallel','serial'] if repeat%2==0 else ['serial','parallel']):
            detector._pool = parallel if mode=='parallel' else inline
            detector.stage_times.reset()
            started = time.perf_counter()
            values = [repr(detector.detect_regions(frame)) for frame in frames]
            elapsed = time.perf_counter()-started
            assert values == baseline, 'Scheduling changed model detections'
            results.append(dict(repeat=repeat, mode=mode, frames=len(frames), seconds=elapsed,
                                stage_seconds=detector.stage_times.snapshot()))
    detector._pool = parallel
    report = dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), frame_indices=sorted(selected),
        outputs_sha256=hashlib.sha256(repr(baseline).encode()).hexdigest(), outputs_equal=True, runs=results,
        notes=['Only quality primary task scheduling changes; nested detail executor stays unchanged',
               'Same real full-frame discovery pixels and warm models; no tracking, AV, lookahead or live backlog test',
               'A microbenchmark result is not an end-to-end optimization result or RTX prediction'])
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([dict(mode=r['mode'],seconds=r['seconds']) for r in results]))
