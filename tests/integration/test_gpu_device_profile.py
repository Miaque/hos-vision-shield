import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from vision_shield.config import Settings
from vision_shield.detection import load_detector, device_profile
from vision_shield.detection.egoblur import _HalfPlate
from vision_shield.redaction import redact


@pytest.mark.skipif(os.getenv('RUN_GPU_TESTS') != '1', reason='requires actual CUDA model traces')
def test_real_parallel_primary_profile_is_bounded_and_keeps_masks(tmp_path, monkeypatch):
    detector = load_detector(Settings(_env_file=None, inference_device='cuda', detection_backend='quality_yolo').model_dump())
    if not isinstance(detector.primary.models[1], _HalfPlate):
        detector.primary.models[1] = _HalfPlate(detector.primary.models[1])
    manifest = json.loads(Path('tests/fixtures/quality-cases.json').read_text('utf-8'))
    case = next(row for row in manifest['cases'] if row['id'] == 'image-7')
    pixels = np.array(Image.open(Path(manifest['local_source']) / case['file']).convert('RGB'))
    expected = redact(pixels, detector.detect(pixels))
    monkeypatch.setattr(device_profile, 'CHECKPOINTS', (2, 4, 6))
    events = []
    sampler = detector.primary.device_samples = device_profile.DeviceSamples(tmp_path, events.append)
    for _ in range(7):
        np.testing.assert_array_equal(redact(pixels, detector.detect(pixels)), expected)
    sampler.finish()
    assert len(events) == 4 and events[-1]['device_profile']['state'] == 'complete'
    assert [sample['frame'] for sample in sampler.samples] == [2, 4, 6]
    assert len(list(tmp_path.glob('plate-profile-*.json'))) == 3
    for sample in sampler.samples:
        assert sample['primary_thread_id'] != threading.get_native_id()
        assert sample['kernel_count'] > 0 and sample['gpu_busy_ms'] > 0
        assert sample['preceding_unprofiled_forward_ms'] > 0
        assert sample['parameter_dtypes'] == ['torch.float16']
        assert sample['input_shape'] == [3, 360, 1280] and sample['input_device'].startswith('cuda:')
        trace = json.loads((tmp_path / f"plate-profile-{sample['frame']}.json").read_text('utf-8'))
        for name, value in device_profile.summarize(trace).items():
            assert sample[name] == value
