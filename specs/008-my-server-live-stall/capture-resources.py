"""Five-second aggregate resource samples for the authorized test containers/GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time


def capture(path, seconds):
    deadline = time.monotonic() + seconds
    containers = ['vision-shield-test-stream-worker-' + str(n) for n in (1, 2, 3)]
    containers += ['vision-shield-test-cuda-mps-1', 'vision-shield-test-mediamtx-1']
    with path.open('w') as output:
        while time.monotonic() < deadline:
            started = time.monotonic()
            sample = dict(unix=time.time(), monotonic=started)
            for name, command in [
                ('gpu', ['nvidia-smi', '-i', 'GPU-ec9780c6-4e06-f44a-8491-e30f0ab74da9',
                         '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                         '--format=csv,noheader,nounits']),
                ('containers', ['docker', 'stats', '--no-stream', '--format', '{{json .}}', *containers]),
                ('gpu_apps', ['nvidia-smi', '-i', 'GPU-ec9780c6-4e06-f44a-8491-e30f0ab74da9',
                              '--query-compute-apps=pid,used_gpu_memory,process_name', '--format=csv,noheader,nounits']),
            ]:
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=4)
                    sample[name] = dict(code=result.returncode, output=result.stdout[:4096])
                except subprocess.TimeoutExpired:
                    sample[name] = dict(timeout=True)
            output.write(json.dumps(sample) + '\n')
            output.flush()
            time.sleep(max(0, min(5 - (time.monotonic() - started), deadline - time.monotonic())))


if __name__ == '__main__':
    assert len(sys.argv) == 3
    seconds = float(sys.argv[2])
    assert 0 < seconds <= 600
    capture(Path(sys.argv[1]), seconds)
