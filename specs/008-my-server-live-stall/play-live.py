"""Bounded FFplay session with wall-clock status evidence and reconnects after EOF."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Timer
import time


def status(line):
    match = re.search(r'^\s*([\d.nan-]+)\s+(A-V|M-V):\s*([\d.-]+)\s+fd=\s*(\d+)', line)
    if not match:
        return None
    result = dict(position=match[1], clock=match[2], difference=float(match[3]), dropped=int(match[4]))
    for field, name in [('aq', 'audio_queue_kb'), ('vq', 'video_queue_kb')]:
        queued = re.search(r'\b' + field + r'=\s*(\d+)KB', line)
        if queued:
            result[name] = int(queued[1])
    return result


def play(binary, url, directory, seconds):
    directory.mkdir(parents=True, exist_ok=True)
    deadline, attempt, written = time.monotonic() + seconds, 0, 0
    with (directory / 'player.jsonl').open('w', encoding='utf-8') as output:
        def record(kind, **values):
            nonlocal written
            line = json.dumps(dict(kind=kind, attempt=attempt, unix=time.time(), monotonic=time.monotonic(), **values)) + '\n'
            written += len(line.encode())
            assert written < 32 * 1024 * 1024, 'player metadata budget exceeded'
            output.write(line)
            output.flush()

        while time.monotonic() < deadline:
            attempt += 1
            process = subprocess.Popen([binary, '-hide_banner', '-loglevel', 'info', '-stats', '-noframedrop',
                '-noinfbuf', '-autoexit', '-rtsp_transport', 'tcp', '-window_title', 'my-server-final-rfm01', url],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            timer = Timer(max(.01, deadline - time.monotonic()), process.terminate)
            timer.start()
            record('open', pid=process.pid, render_driver=os.environ.get('SDL_RENDER_DRIVER'))
            try:
                line = bytearray()
                while chunk := process.stderr.read(1):
                    if chunk in (b'\r', b'\n'):
                        text = line.decode('utf-8', errors='replace')
                        parsed = status(text)
                        if parsed:
                            record('status', **parsed)
                        elif text:
                            record('message', text=text[:512])
                        line.clear()
                    else:
                        line.extend(chunk)
                        assert len(line) < 8192
                record('exit', code=process.wait())
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    assert status('  1.20 A-V: -0.031 fd=   0 aq= 0KB')['difference'] == -.031
    assert status('  1.20 M-V: 0.100 fd= 2 vq= 0KB')['dropped'] == 2
    assert status('not playback progress') is None
    assert status('1.2 A-V: .1 fd=0 aq= 2KB vq= 32KB')['video_queue_kb'] == 32
    if len(sys.argv) > 1:
        play(sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4]))
