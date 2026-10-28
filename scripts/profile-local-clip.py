"""Isolated fixed-clip diagnosis; no service connection or task mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from threading import Event
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vision_shield.config import Settings
from vision_shield.media import child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--mode', choices=('host', 'local-face'), required=True)
    args = parser.parse_args()
    with args.video.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Settings(_env_file=None, debug=True, detection_backend='quality_yolo', inference_device='cuda',
        face_model_path=args.models / 'yunet_face.onnx', plate_model_path=args.models / 'hyperlpr_plate.onnx',
        egoblur_face_model_path=args.models / 'face.jit', egoblur_plate_model_path=args.models / 'license_plate.jit').model_dump(mode='json')
    config.update(detection_interval=3, tracking='flow', redaction_mode='solid')
    events, last = [], -100

    def emit(event):
        nonlocal last
        events.append(event)
        count = event.get('stats', {}).get('processed_frames')
        if count is not None and count - last >= 100:
            print(f'processed_frames={count}', flush=True)
            last = count

    child.emit = emit
    started = time.monotonic()
    code, _ = child.process(dict(kind='video', source=str(args.video.resolve()), directory=str(args.output.resolve()),
                                 config=config, profile_detection=args.mode), Event())
    report = dict(source_sha256=digest, diagnostic_mode=args.mode, exit_code=code,
                  elapsed_seconds=time.monotonic() - started, events=events)
    (args.output / 'observations.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    print(f'exit_code={code} diagnostic_seconds={report["elapsed_seconds"]:.3f}', flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
