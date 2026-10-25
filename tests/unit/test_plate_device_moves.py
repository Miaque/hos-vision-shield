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
        boxes = pixels[:, None].expand(-1, 4)
        keep = (pixels >= 0) & (pixels < 2)
        selected, scores, levels = boxes[keep], result[keep], indices[keep]
        mask = boxes > 0.25
        positions = torch.nonzero(mask)
        return (result, pixels.to(device="cpu", copy=True), selected, scores, levels,
                positions, boxes[mask], boxes[mask])


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


@torch.jit.script
def shared_masks(boxes: torch.Tensor, scores: torch.Tensor, levels: torch.Tensor, matrix: torch.Tensor):
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    a, b, c = boxes[keep], scores[keep], levels[keep]
    mask = matrix > 0.25
    positions = torch.nonzero(mask)
    expanded = boxes[:, None, :].expand(-1, matrix.size(1), -1)
    return a, b, c, positions, expanded[mask], matrix[mask]


@pytest.mark.parametrize('count', [0, 1, 17])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_shared_mask_indices_preserve_filter_order_and_values(count, dtype):
    from vision_shield.detection.egoblur import _shared_mask_indices

    before = str(shared_masks.graph)
    candidate = torch._C._create_function_from_graph('shared_mask_indices', _shared_mask_indices(shared_masks.graph.copy()))
    generator = torch.Generator().manual_seed(123)
    boxes = torch.randn(count, 4, generator=generator, dtype=dtype)
    scores = torch.randn(count, generator=generator, dtype=dtype)
    levels = torch.arange(count)
    matrix = torch.randn(3, count, generator=generator, dtype=dtype).t()
    for values in (matrix, matrix * 0, matrix * 0 + 1):
        inputs = (boxes, scores, levels, values)
        with torch.jit.optimized_execution(False):
            expected, actual = shared_masks(*inputs), candidate(*inputs)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert str(shared_masks.graph) == before
    assert len(candidate.graph.findAllNodes('aten::nonzero')) == 2


def test_shared_mask_indices_reject_unrecognized_filter_graph():
    from vision_shield.detection.egoblur import _shared_mask_indices

    with pytest.raises(RuntimeError, match='shared-mask graph'):
        _shared_mask_indices(torch.jit.trace(torch.nn.Identity(), torch.ones(3)).graph.copy())


def test_shared_mask_indices_compute_nonzero_once_per_mask():
    from vision_shield.detection.egoblur import _shared_mask_indices

    candidate = torch._C._create_function_from_graph('shared_mask_count', _shared_mask_indices(shared_masks.graph.copy()))
    inputs = (torch.randn(17, 4), torch.randn(17), torch.arange(17), torch.randn(17, 3))
    counts = []
    for function in (shared_masks, candidate):
        with torch.jit.optimized_execution(False), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            function(*inputs)
        counts.append(sum(event.count for event in profiler.key_averages() if event.key == 'aten::nonzero'))
    assert counts == [6, 2]
