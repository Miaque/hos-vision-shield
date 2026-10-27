"""NVDEC frames stay on device through color conversion, masking and NVENC input."""

import math

import av
from av.codec.hwaccel import HWAccel
from av.video.frame import CudaContext
import numpy as np
import torch
import torch.nn.functional as F

from vision_shield.detection.cuda_pixels import resize
from vision_shield.detection.timing import measure


def hwaccel(device):
    # ORT may already own the primary context with different scheduling flags.
    return HWAccel("cuda", device=str(device), allow_software_fallback=False, is_hw_owned=True,
                   options={"primary_ctx": "0", "current_ctx": "1"})


def rgb_pixels(y, uv, space, full):
    coefficients = [117489, 138438, -13975, -34925] if space == 1 else [104597, 132201, -25675, -53279]
    if full:
        coefficients = [int(value * 224 / 255) for value in coefficients]
    cy = 65536 if full else 65536 * 255 // 219
    yc = (cy * 8192 + 32768) >> 16
    vr, ub, ug, vg = [(value * 8192 + 32768) >> 16 for value in coefficients]
    chroma = uv.int()
    # Match swscale's separate 13-bit high-word products, including negative rounding.
    yy = (((y.int() * 8 - (0 if full else 128)) * yc) >> 16).view(y.shape[0] // 2, 2, y.shape[1] // 2, 2)
    u, v = chroma[:, :, 0] * 8 - 1024, chroma[:, :, 1] * 8 - 1024
    red = yy + ((v * vr) >> 16)[:, None, :, None]
    green = yy + (((u * ug) >> 16) + ((v * vg) >> 16))[:, None, :, None]
    blue = yy + ((u * ub) >> 16)[:, None, :, None]
    return torch.stack((red, green, blue), -1).clamp(0, 255).byte().reshape(*y.shape, 3)


def gray_pixels(pixels):
    values = pixels.int()
    return ((values[:, :, 0] * 9798 + values[:, :, 1] * 19235 + values[:, :, 2] * 3735 + 16384) >> 15).byte()


def masked_planes(y, uv, pixels, boxes, mode, space, full):
    height, width = y.shape
    # NVENC expects the chroma plane at the luma-pitch offset in the same allocation.
    storage = torch.empty((height * 3 // 2, width), dtype=torch.uint8, device=y.device)
    output_y, output_uv = storage[:height], storage[height:].view(height // 2, width // 2, 2)
    output_y.copy_(y)
    output_uv.copy_(uv)
    for x1, y1, x2, y2 in boxes:
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("Non-finite detection coordinates")
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if right <= left or bottom <= top:
            continue
        chroma = output_uv[top // 2:(bottom + 1) // 2, left // 2:(right + 1) // 2]
        if mode == "solid" or min(right - left, bottom - top) < 2:
            output_y[top:bottom, left:right] = 64 if full else 71
            chroma[:] = 128
        else:
            small = resize(pixels[top:bottom, left:right],
                           (max(1, min(8, (right - left) // 12)), max(1, min(8, (bottom - top) // 12))), "area")
            mosaic = F.interpolate(small.permute(2, 0, 1).unsqueeze(0).float(), (bottom - top, right - left),
                                   mode="nearest")[0]
            red, green, blue = mosaic
            cr, cb = (0.2126, 0.0722) if space == 1 else (0.299, 0.114)
            luma = red * cr + green * (1 - cr - cb) + blue * cb
            color = torch.stack(((blue - luma) / (2 * (1 - cb)), (red - luma) / (2 * (1 - cr))))
            output_y[top:bottom, left:right] = (luma if full else luma * (219 / 255) + 16).round().clamp(0, 255).byte()
            color = color if full else color * (224 / 255)
            chroma[:] = F.interpolate(color.unsqueeze(0), tuple(chroma.shape[:2]), mode="area")[0].permute(1, 2, 0).add(128).round().clamp(0, 255).byte()
    return output_y, output_uv


class CUDAFrames:
    def __init__(self):
        self.device = torch.cuda.current_device()
        torch.empty(0, device=torch.device("cuda", self.device))
        self.context = CudaContext(device_id=self.device, primary_ctx=False, current_ctx=True)
        self.color = None

    def masked(self, frame, redactor, *, defer=False):
        if frame.format.name != "cuda" or frame.sw_format.name != "nv12":
            raise ValueError("Expected NVDEC CUDA NV12 frame")
        if frame.rotation % 90:
            raise ValueError("Unsupported video orientation")
        if frame.colorspace not in {0, 1, 2, 3, 5, 6}:
            raise ValueError("Unsupported NVDEC color matrix")
        color = (frame.colorspace == 1, frame.color_range == 2)
        if self.color is not None and color != self.color:
            raise ValueError("Video color properties changed")
        self.color = color
        with measure(redactor, "explicit-sync"):
            torch.cuda.synchronize(self.device)
        y, uv = [torch.from_dlpack(plane.__dlpack__()) for plane in frame.planes]
        if frame.rotation:
            y, uv = [torch.rot90(plane, frame.rotation // 90, (0, 1)) for plane in (y, uv)]
        full = frame.color_range == 2
        with measure(redactor, "color-convert"):
            pixels = rgb_pixels(y, uv, frame.colorspace, full)
        height, width = pixels.shape[:2]
        rows = torch.tensor(np.linspace(0, height - 1, 36, dtype=int), device=pixels.device)
        cols = torch.tensor(np.linspace(0, width - 1, 64, dtype=int), device=pixels.device)
        with measure(redactor, "gray-copy"):
            thumbnail = pixels[rows[:, None], cols[None, :]].float().cpu().numpy()
            gray = gray_pixels(pixels).cpu().numpy()
        # shortcut: template matching still uses CPU gray pixels; move it only after equivalent tracking is verified.
        boxes = redactor.regions(pixels, float(frame.pts * frame.time_base),
                                 thumbnail=thumbnail, gray=gray)
        frame_number = redactor.frame_index
        def render(extra):
            recorder = getattr(redactor.stage_times, 'record_masks', None)
            if defer and recorder:
                from vision_shield.detection.host_profile import scope

                with scope(frame=frame_number):
                    recorder(pixels.shape, [*boxes, *extra])
            with measure(redactor, "mask-gpu"):
                output_y, output_uv = masked_planes(y, uv, pixels, [*boxes, *extra], redactor.mode, frame.colorspace, full)
            with measure(redactor, "explicit-sync"):
                torch.cuda.current_stream(self.device).synchronize()
            output = av.VideoFrame.from_dlpack((output_y, output_uv), format="nv12", cuda_context=self.context)
            output.pts, output.time_base = frame.pts, frame.time_base
            output.color_range = 2 if full else 1
            output.colorspace, output.color_trc, output.color_primaries = frame.colorspace, frame.color_trc, frame.color_primaries
            return output

        if defer:
            from vision_shield.media.plate_lookahead import PendingMask

            moment = float(frame.pts * frame.time_base)
            plates = [t.box for t in redactor.history if t.kind == "plate"]
            confirmed = [t.box for t in redactor.history if t.kind == "plate" and t.seen == moment]
            return PendingMask(moment, gray, plates, confirmed, render, redactor.scene_reset)
        return render(())
