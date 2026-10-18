"""Local-only robot comparison. Counts are detections, never ground-truth recall."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from vision_shield.config import Settings
from vision_shield.detection.egoblur import Detector
from vision_shield.redaction import FrameRedactor, redact


def evaluate(source: Path, output: Path) -> None:
    if not source.is_dir() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Use an existing source directory and a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = Settings()
    detector = Detector(config.face_model_path, config.plate_model_path,
                        config.detection_max_side, config.detection_box_padding)
    report = dict(source_files=[], images=[], video_frames=[],
                  parameters=dict(max_side=detector.max_side, box_padding=detector.box_padding,
                                  score_threshold=0.1, nms_iou=0.3, hold_seconds=0.2),
                  review="No human labels; detection counts are not accuracy")

    def compare(pixels, name, temporal=None, moment=None):
        old_side, old_padding = detector.max_side, detector.box_padding
        try:
            detector.max_side, detector.box_padding = 640, 0
            start = time.perf_counter()
            before = detector.detect(pixels)
            baseline_seconds = time.perf_counter() - start
        finally:
            detector.max_side, detector.box_padding = old_side, old_padding
        start = time.perf_counter()
        after = detector.detect(pixels)
        seconds = time.perf_counter() - start
        final = redact(pixels, after)
        if temporal is not None:
            # Replay actual detections, avoiding a second expensive model call.
            class Observed:
                def detect(self, image):
                    return after
            temporal.detector = Observed()
            final = temporal.apply(pixels, moment)
        Image.fromarray(final).save(output / f"{name}-redacted.jpg", quality=95)
        panels = []
        for label, boxes in [("before-640", before), ("after", after)]:
            preview = Image.fromarray(pixels)
            draw = ImageDraw.Draw(preview)
            for box in boxes:
                draw.rectangle(box, outline="red", width=3)
            preview.thumbnail((960, 540))
            panels.append(preview)
        canvas = Image.new("RGB", (1920, max(p.height for p in panels) + 24), "#222222")
        draw = ImageDraw.Draw(canvas)
        for index, (label, panel) in enumerate(zip(["before-640", "after"], panels, strict=True)):
            canvas.paste(panel, (index * 960, 24))
            draw.text((index * 960 + 8, 5), label, fill="white")
        canvas.save(output / f"{name}-compare.jpg", quality=95)
        return dict(name=name, dimensions=[pixels.shape[1], pixels.shape[0]],
                    baseline_boxes=before, boxes=after,
                    baseline_seconds=baseline_seconds, seconds=seconds)

    for index, path in enumerate(sorted(source.iterdir())):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".mp4"}:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["source_files"].append(dict(index=index, name=path.name, sha256=digest))
        if path.suffix.lower() != ".mp4":
            with Image.open(path) as image:
                pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"))
            report["images"].append(compare(pixels, f"image-{index}"))
        else:
            redactor = FrameRedactor(detector)
            with av.open(str(path)) as reader:
                # Consecutive frames test a blur gap; this is not full-video processing.
                for frame_index, frame in enumerate(reader.decode(video=0)):
                    if frame_index == 3:
                        break
                    if frame.pts is None:
                        raise ValueError("Robot sample has a missing frame timestamp")
                    pixels = frame.to_ndarray(format="rgb24")
                    row = compare(pixels, f"video-{index}-frame-{frame_index}", redactor,
                                  float(frame.pts * frame.time_base))
                    row.update(pts=frame.pts, time_base=str(frame.time_base))
                    report["video_frames"].append(row)
        (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        print(f"Evaluated sample index {index}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/validation/robot/evaluated"))
    args = parser.parse_args()
    evaluate(args.source, args.output)
