import os

import pytest
import torch
import torchvision

from vision_shield.detection import device_profile
from vision_shield.detection.egoblur import _HalfPlate


@pytest.mark.skipif(os.getenv('RUN_GPU_TESTS') != '1', reason='requires original CUDA plate graph')
def test_actual_anchor_reuse_removes_uploads_and_preserves_all_model_outputs(tmp_path, monkeypatch):
    cached = _HalfPlate(torch.jit.load('models/license_plate.jit', map_location='cpu').cuda().eval())

    class Original(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cached.model

        def forward(self, pixels):
            with torch.autocast('cuda', dtype=torch.float16):
                boxes, labels, scores, size = self.model(pixels)
            return boxes.float(), labels, scores.float(), size

    original = Original()
    assert cached.model.model.model.pixel_mean.dtype == cached.model.model.model.pixel_std.dtype == torch.float32
    assert all(parameter.dtype == torch.float16 for parameter in cached.parameters())
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for index, (height, width) in enumerate([(360, 1280), (319, 321), (320, 321), (321, 319), (320, 320), (64, 64), (360, 1280)]):
            pixels = torch.arange(3 * height * width, device='cuda').reshape(3, height, width).add_(index).to(torch.uint8)
            expected, actual = original(pixels), cached(pixels)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)), (height, width)
            assert actual[-1].device.type == 'cpu' and actual[-1].tolist() == [height, width]
        assert len(cached.anchors) == 4
        snapshot = {key: tuple(value.clone() for value in values) for key, values in cached.anchors.items()}
        monkeypatch.setattr(device_profile, 'CHECKPOINTS', (1,))
        profiles = []
        for name, model in [('original', original), ('cached', cached)]:
            directory = tmp_path / name
            directory.mkdir()
            sampler = device_profile.DeviceSamples(directory, lambda event: None)
            sampler.forward(model, pixels, global_view=True)
            profiles.append(sampler.samples[0])
        before, after = profiles
        assert before['HtoD']['calls'] - after['HtoD']['calls'] >= 10
        assert before['synchronization_api_events'] - after['synchronization_api_events'] >= 10
        assert after['kernel_count'] < before['kernel_count']
        assert all(torch.equal(a, b) for key, values in cached.anchors.items()
                   for a, b in zip(values, snapshot[key], strict=True)), 'RPN must never mutate cached anchors'
