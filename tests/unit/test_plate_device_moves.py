import pytest
import torch


@torch.jit.script
def move_device_like(source: torch.Tensor, target: torch.Tensor):
    return source.to(target.device)


class FPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, eps=0)
        with torch.no_grad():
            self.conv.weight.fill_(0.25)
            self.bn.weight.fill_(2)
            self.bn.bias.fill_(0.5)
            self.bn.running_mean.fill_(0.125)
            self.bn.running_var.fill_(0.25)

    def forward(self, pixels):
        return self.bn(self.conv(pixels.reshape(-1, 1, 1, 1))).flatten()


class PlateCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FPN()


class PlateMoves(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PlateCore()

    def forward(self, pixels):
        indices = torch.arange(pixels.shape[0], dtype=pixels.dtype, device="cpu")
        result = self.model.backbone(pixels)
        for _ in range(17):
            result = result + move_device_like(indices, pixels)
        return result, pixels.to(device="cpu", copy=True)


class PlateWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PlateMoves()

    def forward(self, pixels):
        return self.model(pixels)


@pytest.mark.parametrize("size", [0, 1, 17])
def test_frozen_plate_keeps_outputs_and_transfer_contract(size):
    from vision_shield.detection.egoblur import _plate_forward

    original = torch.jit.trace(PlateWrapper().eval(), torch.ones(3), check_trace=False)
    original_graph = str(original.inlined_graph)
    candidate = _plate_forward(original)
    assert not candidate.graph.findAllNodes("aten::batch_norm")
    assert len(original.inlined_graph.findAllNodes("aten::batch_norm")) == 1
    transfers = candidate.graph.findAllNodes("aten::to")
    assert sum(node.inputsAt(3).toIValue() is True for node in transfers) == 17
    assert str(transfers[-1].inputsAt(3).toIValue()) == "cpu"
    assert transfers[-1].inputsAt(5).toIValue() is False
    assert transfers[-1].inputsAt(6).toIValue() is True
    assert str(original.inlined_graph) == original_graph

    pixels = torch.arange(size, dtype=torch.float32)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected = original(pixels)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            actual = candidate(original, pixels)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert all(event.name != "move_device_like" for event in profiler.events())
    assert torch.equal(pixels, torch.arange(size, dtype=torch.float32))


def test_unrecognized_device_graph_fails_closed():
    from vision_shield.detection.egoblur import _plate_forward

    with pytest.raises(RuntimeError, match="Gen1 .*graph"):
        _plate_forward(torch.jit.trace(torch.nn.Identity(), torch.ones(3)))


def test_explicit_cpu_backbone_is_folded_without_mutating_live_model(monkeypatch):
    from vision_shield.detection.egoblur import _plate_forward

    original = torch.jit.trace(PlateWrapper().eval(), torch.ones(3), check_trace=False)
    backbone = torch.jit.trace(FPN().eval(), torch.ones(3), check_trace=False)
    before = str(original.inlined_graph)
    freeze, folded = torch.jit.freeze, []

    def observe(module):
        folded.append(module)
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        return freeze(module)

    monkeypatch.setattr(torch.jit, "freeze", observe)
    candidate = _plate_forward(original, backbone)
    assert folded == [backbone]
    assert str(original.inlined_graph) == before
    assert not candidate.graph.findAllNodes("aten::batch_norm")
    for size in (0, 1, 17):
        pixels = torch.arange(size, dtype=torch.float32)
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            expected, actual = original(pixels), candidate(original, pixels)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    with pytest.raises(RuntimeError, match="constant dtype"):
        _plate_forward(original, backbone.double())
    assert str(original.inlined_graph) == before
