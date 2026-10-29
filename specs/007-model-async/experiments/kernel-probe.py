"""Separate CUDA samples on fixed real pixels; sampled timings are not throughput."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import av
import cv2
import torch

from vision_shield.config import Settings
from vision_shield.detection import load_detector
from vision_shield.detection.host_profile import ProfileTimes, instrument_candidates, scope
from vision_shield.detection.cuda_pixels import upload


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    with av.open(str(args.source)) as reader:
        pixels = next(reader.decode(video=0)).to_ndarray(format='rgb24')
    detector = load_detector(Settings(_env_file=None, debug=True, detection_backend='quality_yolo',
                                      inference_device='cuda').model_dump(mode='json'))
    results = []
    for group in ('primary','detail'):
        model = getattr(detector, group)
        for view, image in [('full', pixels), ('resized-320', cv2.resize(pixels, (320,240)))]:
            stage = f'forward-{group}-plate'
            directory = args.directory/f'{group}-{view}'
            collector = ProfileTimes(directory, device_stage=stage, checkpoints=(6,7,8))
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(model, 'stage_times', collector))
                    instrument_candidates(model, group, collector, stack=stack)
                    gpu = upload(image, detector.primary.device)
                    values = []
                    for i in range(8):
                        with scope(frame=1, purpose='fixed-pixel-kernel-probe', variant=view):
                            output = model.candidates(gpu, kind='plate')
                            values.append(repr(output))
                assert len(set(values)) == 1, 'Repeated identical pixels changed detections'
                report = collector.report(8)
                results.append(dict(model=group, view=view, shape=list(image.shape), report=report))
            finally:
                collector.close()
    report = dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), probes=results,
        notes=['One fixed decoded frame, independently timed serialized models, not the concurrent rfm critical path',
               '320 view is an aspect-preserving resize, not an actual confirmation ROI',
               'CUDA profiler can add synchronization and setup; never use this run for FPS or observer overhead'])
    (args.directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(gpu=report['gpu'], probes=len(results))))
