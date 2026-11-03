import pytest
import torch

from vision_shield.detection.plate_graph import replace_backbone


class Backbone(torch.nn.Module):
    def forward(self, pixels):
        return pixels + 1, pixels + 2, pixels + 3, pixels + 4, pixels + 5


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()

    def forward(self, pixels):
        a, b, c, d, e = self.backbone(pixels)
        return a + e, b + d, c


def test_replaces_only_backbone_preserving_dynamic_head_and_original_graph():
    model = torch.jit.script(Wrapper())
    library = torch.library.Library('vision_shield_plate', 'IMPL', 'CPU')
    calls = []
    library.impl('backbone', lambda pixels, owner: calls.append(owner) or Backbone()(pixels))
    graph = replace_backbone(model.forward.graph.copy(), 123)
    forward = torch._C._create_function_from_graph('plate_graph_test', graph)
    assert 'vision_shield_plate::backbone' not in str(model.graph)
    for n in (0, 2, 7):
        pixels = torch.arange(n)
        assert all(torch.equal(a, b) for a, b in zip(model(pixels), forward(model, pixels), strict=True))
    assert calls == [123, 123, 123]


def test_unrecognized_backbone_fails_closed():
    with pytest.raises(RuntimeError, match='backbone graph'):
        replace_backbone(torch.jit.trace(torch.nn.Identity(), torch.ones(2)).graph.copy(), 123)
