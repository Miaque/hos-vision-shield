import torch

from vision_shield.detection.egoblur import _HalfPlate


class Backbone(torch.nn.Module):
    def forward(self, pixels):
        return pixels[:, :1, ::4, ::4], pixels[:, :1, ::8, ::8], pixels[:, :1, ::16, ::16], pixels[:, :1, ::32, ::32], pixels[:, :1, ::64, ::64]


class Anchors(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, a, b, c, d, e):
        self.calls += 1
        return (torch.ones((a.shape[2], 4)) * a.shape[3], torch.ones((b.shape[2], 4)) * b.shape[3],
                torch.ones((c.shape[2], 4)) * c.shape[3], torch.ones((d.shape[2], 4)) * d.shape[3],
                torch.ones((e.shape[2], 4)) * e.shape[3])


class RPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_generator = Anchors()

    def forward(self, a, b, c, d, e, size):
        anchors = self.anchor_generator(a, b, c, d, e)
        return torch.cat(anchors) + a.mean() + size[0]


class ROI(torch.nn.Module):
    def forward(self, a, proposals, b, c, d, size):
        return proposals, torch.ones(len(proposals), dtype=torch.int64), proposals[:, 0] / size[1]


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pixel_mean', torch.zeros((3, 1, 1)))
        self.register_buffer('pixel_std', torch.ones((3, 1, 1)))
        self.backbone, self.proposal_generator, self.roi_heads = Backbone(), RPN(), ROI()


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Core()

    def forward(self, pixels):
        size = torch.tensor([pixels.shape[1], pixels.shape[2]])
        normalized = (pixels - self.model.pixel_mean) / self.model.pixel_std
        features = self.model.backbone(torch.nn.functional.pad(
            normalized, (0, (-pixels.shape[2]) % 32, 0, (-pixels.shape[1]) % 32)).unsqueeze(0).contiguous())
        proposals = self.model.proposal_generator(*features, size)
        boxes, labels, scores = self.model.roi_heads(features[0], proposals, features[1], features[2], features[3], size)
        return boxes, labels, scores, size


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wrapper()

    def forward(self, pixels):
        return self.model(pixels)


def test_reused_anchors_preserve_new_pixels_and_original_unpadded_size():
    original = torch.jit.script(Model().eval())
    cached = _HalfPlate(torch.jit.script(Model().eval()))
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for height, width, value in [(31, 32, 1), (32, 31, 2), (31, 31, 3), (64, 32, 4)]:
            pixels = torch.full((3, height, width), value, dtype=torch.uint8)
            expected, actual = original(pixels), cached(pixels)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
            assert actual[-1].device.type == 'cpu' and actual[-1].tolist() == [height, width]
        assert cached.model.model.model.proposal_generator.anchor_generator.calls == 2


def test_anchor_cache_stays_bounded_and_handles_uncached_shapes():
    original = torch.jit.script(Model().eval())
    cached = _HalfPlate(torch.jit.script(Model().eval()))
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for height in [32, 64, 96, 128, 160, 192, 32]:
            pixels = torch.full((3, height, 32), 7, dtype=torch.uint8)
            expected, actual = original(pixels), cached(pixels)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        assert len(cached.anchors) == 4
        assert cached.model.model.model.proposal_generator.anchor_generator.calls == 6
