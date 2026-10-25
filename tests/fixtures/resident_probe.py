"""Exercise the real child protocol without GPU weights or external media servers."""

import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

from vision_shield import detection
from vision_shield.detection import preflight
from vision_shield.detection.timing import StageTimes
from vision_shield.media import child, images, streams


loads = 0


def load(config):
    global loads
    loads += 1
    return SimpleNamespace(stage_times=StageTimes(), candidate_timings={}, _proposal_candidates=None)


def image(source, target, detector, limit, *, redaction_mode):
    child.emit(dict(observed=dict(pid=os.getpid(), loads=loads, mode=redaction_mode,
                                  previous_seconds=detector.stage_times.snapshot()['forward-primary-face'],
                                  previous_candidates=detector._proposal_candidates)))
    detector.stage_times.seconds['forward-primary-face'] = 3
    detector._proposal_candidates = 'previous task'
    Path(target).write_bytes(b'result')


def stream(source, target, detector, config, stop, emit):
    with socket.socket() as publisher:
        publisher.bind(('127.0.0.1', 0))
        publisher.listen()
        emit(dict(publishing=dict(pid=os.getpid(), loads=loads, source=source,
                                  port=publisher.getsockname()[1])))
        if source == 'retry':
            raise OSError('retryable connection loss')
        if source == 'fatal':
            raise RuntimeError('CUDA failure')
        if source == 'stuck':
            time.sleep(30)
        elif source != 'eof':
            stop.wait()


detection.load_detector = load
preflight.verify_models = lambda config: []
preflight.validate_device = lambda device: None
preflight.run_inference = lambda config, detector=None: dict(device='cpu', loads=loads)
images.process_image = image
streams.process_stream = stream
child.main(persistent='--persistent' in sys.argv)
