"""Bounded direct-source decode; persist clocks and hashes, never pictures."""
import hashlib
import json
from pathlib import Path
import sys
import time

import av


def measure(url, path, seconds):
    assert 0 < seconds <= 600
    deadline = time.monotonic() + seconds
    written = 0
    with path.open('w') as output:
        def record(kind, **data):
            nonlocal written
            line = json.dumps(dict(kind=kind, unix=time.time(), monotonic=time.monotonic(), **data)) + '\n'
            written += len(line.encode())
            assert written < 16*1024*1024
            output.write(line)

        while time.monotonic() < deadline:
            try:
                with av.open(url, options={'rtsp_transport': 'tcp', 'analyzeduration': '100000',
                             'probesize': '32768'}, timeout=(5, 5)) as reader:
                    video = reader.streams.video[0]
                    video.thread_type = 'SLICE'
                    record('open', average_rate=str(video.average_rate), base_rate=str(video.base_rate),
                           time_base=str(video.time_base), width=video.width, height=video.height)
                    for packet in reader.demux(video=0):
                        if time.monotonic() >= deadline:
                            return
                        record('packet', pts=packet.pts, dts=packet.dts, time_base=str(packet.time_base),
                               bytes=packet.size, sha256=hashlib.sha256(bytes(packet)).hexdigest())
                        for frame in packet.decode():
                            pixels = frame.to_ndarray(format='gray')
                            record('frame', pts=frame.pts, time_base=str(frame.time_base),
                                   sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                        output.flush()
                    record('eof')
            except (OSError, av.FFmpegError) as error:
                record('error', type=type(error).__name__)


if __name__ == '__main__':
    measure(sys.argv[1], Path(sys.argv[2]), float(sys.argv[3]))
