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
from vision_shield.detection.quality import _timed_candidates
from vision_shield.media import child, images, streams


loads = 0


def fail_model():
    def candidates(*args, **kwargs):
        raise RuntimeError('private:secret model execution failed')
    _timed_candidates(SimpleNamespace(candidates=candidates), object())


def load(config):
    global loads
    loads += 1
    marker = Path(config['probe_failure_file']) if config.get('probe_failure_file') else None
    if marker and marker.exists():
        fault = marker.read_text('utf-8')
        marker.unlink()
        if fault == 'runtime':
            fail_model()
        if fault == 'device':
            raise preflight.DeviceUnavailableError('模型预检失败：CUDA 不可用')
        raise ValueError('Invalid model configuration')
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
        if source.rsplit('/', 1)[-1] == 'model-once':
            marker = Path(config['model_failure_file'])
            if not marker.exists():
                marker.touch()
                if config.get('probe_failure_file'):
                    Path(config['probe_failure_file']).write_text(config['probe_failure_kind'], 'utf-8')
                fail_model()
        if source == 'retry':
            raise OSError('retryable connection loss')
        if source == 'fatal':
            raise RuntimeError('CUDA failure')
        if source == 'stuck':
            time.sleep(30)
        elif source != 'eof':
            stop.wait()


detection.load_detector = load
def verify(config):
    marker = Path(config['probe_failure_file']) if config.get('probe_failure_file') else None
    if marker and marker.exists() and marker.read_text('utf-8') == 'hash':
        raise preflight.PreflightError('Model hash mismatch')
    return []


preflight.verify_models = verify
preflight.validate_device = lambda device: None
preflight.run_inference = lambda config, detector=None: dict(device='cpu', loads=loads)
images.process_image = image
streams.process_stream = stream
child.main(persistent='--persistent' in sys.argv)
