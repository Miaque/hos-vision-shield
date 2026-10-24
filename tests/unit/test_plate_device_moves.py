import pytest
import torch


@torch.jit.script
def move_device_like(source: torch.Tensor, target: torch.Tensor):
    return source.to(target.device)


class PlateMoves(torch.nn.Module):
    def forward(self, pixels):
        indices = torch.arange(pixels.shape[0], dtype=pixels.dtype, device="cpu")
        result = pixels
        for _ in range(17):
            result = result + move_device_like(indices, pixels)
        return result, pixels.to(device="cpu", copy=True)


@pytest.mark.parametrize("size", [0, 1, 17])
def test_only_internal_device_moves_become_nonblocking(size):
    from vision_shield.detection.egoblur import _async_device_moves

    original = torch.jit.trace(PlateMoves(), torch.ones(3), check_trace=False)
    candidate = _async_device_moves(original)
    transfers = candidate.graph.findAllNodes("aten::to")
    assert sum(node.inputsAt(3).toIValue() is True for node in transfers) == 17
    assert str(transfers[-1].inputsAt(3).toIValue()) == "cpu"
    assert transfers[-1].inputsAt(5).toIValue() is False
    assert transfers[-1].inputsAt(6).toIValue() is True
    assert "move_device_like" in str(original.graph)

    pixels = torch.arange(size, dtype=torch.float32)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        expected = original(pixels)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            actual = candidate(original, pixels)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert all(event.name != "move_device_like" for event in profiler.events())
    assert torch.equal(pixels, torch.arange(size, dtype=torch.float32))


def test_unrecognized_device_graph_fails_closed():
    from vision_shield.detection.egoblur import _async_device_moves

    with pytest.raises(RuntimeError, match="device-transfer graph"):
        _async_device_moves(torch.jit.trace(torch.nn.Identity(), torch.ones(3)))
