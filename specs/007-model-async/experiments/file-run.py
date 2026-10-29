"""Local complete-clip comparison using the real quality/media path; no service access."""
import argparse
import hashlib
import json
from pathlib import Path
from threading import Event
import time

import av
import torch

from vision_shield.config import Settings
from vision_shield.media import child


def media(path):
    video, audio = [], []
    with av.open(str(path)) as reader:
        for frame in reader.decode(video=0):
            video.append((frame.width, frame.height, str(frame.pts*frame.time_base)))
    with av.open(str(path)) as reader:
        for packet in reader.demux(audio=0) if reader.streams.audio else []:
            if packet.size:
                audio.append((str(packet.pts*packet.time_base), str(packet.dts*packet.time_base),
                              hashlib.sha256(bytes(packet)).hexdigest()))
    return video, audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=['normal', 'host', 'async-host'], required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    cfg = Settings(_env_file=None, debug=True, detection_backend='quality_yolo', inference_device='cuda').model_dump(mode='json')
    cfg.update(detection_interval=3, tracking='flow', redaction_mode='solid')
    last, result, largest_event = 0, {}, 0
    def emit(event):
        nonlocal last, largest_event
        largest_event = max(largest_event, len(json.dumps(event).encode()))
        if event.get('stats', {}).get('processed_frames', 0) >= last+100:
            print(json.dumps(event['stats']), flush=True)
            last = event['stats']['processed_frames']
        result.update(event)
    child.emit = emit
    start = time.monotonic()
    code, _ = child.process(dict(kind='video', source=str(args.source.resolve()), directory=str(args.directory.resolve()),
        config=cfg, **({'profile_detection':args.mode} if args.mode!='normal' else {})), Event())
    outcome = dict(mode=args.mode, exit_code=code, total_wall_seconds=time.monotonic()-start,
        gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), largest_event_bytes=largest_event,
        result=result, note='E0 stream probes are inactive in process_video; this file run cannot measure their on/off cost')
    if code == 0:
        original, output = media(args.source), media(args.directory/'result.mp4')
        assert original == output, 'Source/output frame, PTS or AAC content changed'
        assert len(output[0]) == result['stats']['processed_frames'] and result['stats']['dropped_frames']==0
        outcome.update(frames=len(output[0]), audio_packets=len(output[1]), integrity=True,
            output_sha256=hashlib.sha256((args.directory/'result.mp4').read_bytes()).hexdigest())
    (args.directory/'outcome.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in outcome.items() if k!='result'}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
