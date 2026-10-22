"""CUDA image preparation; only small resize coefficients are built on CPU."""

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


def upload(pixels, device):
    if isinstance(pixels, torch.Tensor):
        if (not pixels.is_cuda or pixels.dtype != torch.uint8 or pixels.ndim != 3
                or pixels.shape[2] != 3 or not pixels.numel()
                or pixels.device.index != (device.index if device.index is not None else torch.cuda.current_device())):
            raise ValueError("Expected nonempty CUDA RGB uint8 pixels on the selected device")
        torch.cuda.current_stream(device).synchronize()
        return pixels
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8 or not pixels.size:
        raise ValueError("Expected nonempty RGB uint8 pixels")
    result = torch.from_numpy(np.ascontiguousarray(pixels)).to(device)
    # Both model threads must see the completed upload, including their own streams.
    torch.cuda.current_stream(device).synchronize()
    return result


@lru_cache(maxsize=16)
def _weights(source, target, mode, device):
    scale = source / target
    center = (np.arange(target, dtype=np.float64) + 0.5) * scale
    if mode == "lanczos":
        radius = 3 * max(1, scale)
        left = np.maximum(0, np.trunc(center - radius + 0.5).astype(np.int64))
        right = np.minimum(source, np.trunc(center + radius + 0.5).astype(np.int64))
        indices = left[:, None] + np.arange(int(np.ceil(radius * 2)) + 2)
        distance = (indices - center[:, None] + 0.5) / max(1, scale)
        weights = np.sinc(distance) * np.sinc(distance / 3)
        weights *= (indices < right[:, None]) & (np.abs(distance) < 3)
        weights /= weights.sum(axis=1, keepdims=True)
        # Pillow uses 22-bit coefficients and rounds/clips after each separable pass.
        weights = np.where(weights < 0, np.ceil(weights * 2**22 - 0.5), np.floor(weights * 2**22 + 0.5)) / 2**22
    else:
        left = np.floor(np.arange(target) * scale).astype(np.int64)
        indices = left[:, None] + np.arange(int(np.ceil(scale)) + 1)
        weights = np.maximum(0, np.minimum(indices + 1, (np.arange(target)[:, None] + 1) * scale)
                             - np.maximum(indices, np.arange(target)[:, None] * scale)) / scale
    return (torch.tensor(np.clip(indices, 0, source - 1), device=device),
            torch.tensor(weights, device=device, dtype=torch.float32))


def resize(pixels, size, mode):
    width, height = size
    if pixels.shape[:2] == (height, width):
        return pixels
    if mode == "linear":
        result = F.interpolate(pixels.permute(2, 0, 1).unsqueeze(0).float(), (height, width),
                               mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        return result.round().clamp(0, 255).byte()
    result = pixels
    if pixels.shape[1] != width:
        indices, weights = _weights(pixels.shape[1], width, mode, str(pixels.device))
        # Reduce weighted samples without materializing every tap for every source row.
        result = F.embedding_bag(indices, pixels.permute(1, 0, 2).reshape(pixels.shape[1], -1).float(),
                                 mode="sum", per_sample_weights=weights).reshape(width, pixels.shape[0], 3).permute(1, 0, 2)
        if mode == "lanczos":
            result = (result + 0.5).floor().clamp(0, 255).byte()
    if pixels.shape[0] != height:
        indices, weights = _weights(pixels.shape[0], height, mode, str(pixels.device))
        result = F.embedding_bag(indices, result.reshape(result.shape[0], -1).float(),
                                 mode="sum", per_sample_weights=weights).reshape(height, width, 3)
    return ((result + 0.5).floor() if mode == "lanczos" else result.round()).clamp(0, 255).byte()


def pad(pixels, left, right, top, bottom, value=0):
    return F.pad(pixels.permute(2, 0, 1), (left, right, top, bottom), value=value).permute(1, 2, 0)


def blob(pixels, scale=1):
    result = pixels.permute(2, 0, 1).unsqueeze(0).contiguous().float()
    return result.mul_(scale) if scale != 1 else result


def run(session, tensor, stream):
    stream.synchronize()
    binding = session.io_binding()
    binding.bind_input(session.get_inputs()[0].name, "cuda", tensor.device.index,
                       np.float32, tuple(tensor.shape), tensor.data_ptr())
    for output in session.get_outputs():
        binding.bind_output(output.name, "cpu")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()
