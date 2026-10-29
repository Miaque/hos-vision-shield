"""Real local ORT observation; the zero threshold calibrates capture, not tail severity."""
from contextlib import ExitStack
import json
import os

import numpy as np
import pytest
import torch

from vision_shield.config import Settings
from vision_shield.detection import host_profile
from vision_shield.detection.cuda_detail import Detector


@pytest.mark.skipif(os.getenv('RUN_GPU_TESTS') != '1', reason='requires actual ORT CUDA and CUPTI')
def test_local_ort_trace_captures_actual_cpu_input_cuda_execution(tmp_path, monkeypatch):
    settings = Settings(_env_file=None)
    detector = Detector(settings.face_model_path, settings.plate_model_path)
    pixels = np.zeros((320, 176, 3), np.uint8)
    expected = detector._faces(pixels)
    collector = host_profile.ProfileTimes(tmp_path / 'profile', device_stage='local-face')
    detector.stage_times = collector
    monkeypatch.setattr(host_profile, 'LOCAL_FACE_SLOW_MS', 0)
    try:
        host_profile.instrument_ort_nodes(detector, collector)
        with ExitStack() as stack:
            host_profile.instrument_local_faces(stack, collector)
            with host_profile.scope(frame=1, purpose='local-confirm', local_id='1:1',
                                    local_face=True, local_details={}):
                actual = detector._faces(pixels)
        assert actual is None if expected is None else np.array_equal(actual, expected)
        collector.report(1)
        sample, = collector.device_samples
        assert sample['context']['input_shape'] == [1, 3, 320, 192]
        assert sample['context']['provider'] == 'CUDAExecutionProvider'
        assert sample['kernel_count'] > 0 and sample['HtoD']['bytes'] > 0
        assert sample['ort_top_nodes'] and sample['ort_top_nodes'][0]['provider'] == 'CUDAExecutionProvider'
        assert sample['context']['purpose'] == 'local-confirm'
        assert json.loads((tmp_path / 'profile/local-face-64.json').read_text())['traceEvents']
    finally:
        collector.close()
