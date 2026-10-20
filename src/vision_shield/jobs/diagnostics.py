"""Bounded worker-side samples; the API container cannot observe worker resources."""
import asyncio
import csv
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time

from redis.exceptions import RedisError

SAMPLE_SECONDS = 1
STALE_SECONDS = 5


class ResourceSampler:
    def __init__(self, device, cgroup=None):
        self.device = device
        if cgroup is None:
            cgroup = Path('/sys/fs/cgroup')
            try:
                for line in Path('/proc/self/cgroup').read_text().splitlines():
                    if line.startswith('0::'):
                        cgroup /= line[3:].lstrip('/')
                        break
            except OSError:
                pass
        self.cgroup = cgroup
        self.previous = None

    def sample(self):
        now = time.monotonic()
        window = now - self.previous[0] if self.previous else None
        cpu = dict(scope='cgroup_v2', percent=None, limit_cores=None, unavailable_reason=None)
        usage = None
        try:
            values = dict(line.split() for line in (self.cgroup / 'cpu.stat').read_text().splitlines())
            usage = int(values['usage_usec']) / 1_000_000
            if self.previous and self.previous[1] is not None and window > 0 and usage >= self.previous[1]:
                cpu['percent'] = (usage - self.previous[1]) / window * 100
            else:
                cpu['unavailable_reason'] = 'warming_up'
        except (OSError, ValueError, KeyError):
            cpu['unavailable_reason'] = 'cgroup_v2_unavailable'
        limits = []
        try:
            quota, period = (self.cgroup / 'cpu.max').read_text().split()
            if quota != 'max':
                limits.append(int(quota) / int(period))
            spans = (self.cgroup / 'cpuset.cpus.effective').read_text().strip().split(',')
            if spans != ['']:
                limits.append(sum(int(span.split('-')[-1]) - int(span.split('-')[0]) + 1 for span in spans))
        except (OSError, ValueError, ZeroDivisionError):
            pass
        cpu['limit_cores'] = min(limits) if limits else None
        self.previous = now, usage
        memory = dict(scope='cgroup_v2', used_bytes=None, limit_bytes=None, unavailable_reason=None)
        try:
            memory['used_bytes'] = int((self.cgroup / 'memory.current').read_text())
            limit = (self.cgroup / 'memory.max').read_text().strip()
            memory['limit_bytes'] = None if limit == 'max' else int(limit)
        except (OSError, ValueError):
            memory['unavailable_reason'] = 'cgroup_v2_unavailable'
        return dict(sample_window_seconds=window, cpu=cpu, memory=memory, gpu=self.gpu())

    def gpu(self):
        # Device counters include other processes on this card; they are not per-project GPU usage.
        result = dict(scope='device', devices=[], unavailable_reason=None)
        device_id = os.environ.get('GPU_DEVICE_ID')
        if self.device != 'cuda' or not device_id:
            result['unavailable_reason'] = 'cpu_inference' if self.device != 'cuda' else 'gpu_selection_unavailable'
            return result
        command = ['nvidia-smi', f'--id={device_id}',
                   '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw',
                   '--format=csv,noheader,nounits']
        try:
            output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for row in csv.reader(output.stdout.splitlines(), skipinitialspace=True):
                uuid, name, utilization, used, total, power = row
                def number(value):
                    if value.strip() in {'[N/A]', 'N/A', '[Not Supported]'}:
                        return None
                    parsed = float(value)
                    if not math.isfinite(parsed) or parsed < 0:
                        raise ValueError('Invalid GPU counter')
                    return parsed
                used, total = number(used), number(total)
                result['devices'].append(dict(uuid=uuid.strip(), name=name.strip(),
                    utilization_percent=number(utilization),
                    memory_used_bytes=round(used * 1024**2) if used is not None else None,
                    memory_total_bytes=round(total * 1024**2) if total is not None else None,
                    power_watts=number(power)))
            if len(result['devices']) != 1:
                raise ValueError('Expected one configured GPU')
        except FileNotFoundError:
            result.update(devices=[], unavailable_reason='nvidia_smi_unavailable')
        except subprocess.TimeoutExpired:
            result.update(devices=[], unavailable_reason='gpu_query_timeout')
        except (OSError, subprocess.SubprocessError, ValueError):
            result.update(devices=[], unavailable_reason='gpu_query_failed')
        return result


async def observe_resources(client, pool, worker_id, device):
    sampler = ResourceSampler(device)
    while True:
        try:
            sample = await asyncio.to_thread(sampler.sample)
            seconds, micros = await client.time()
            sample['sampled_at_unix'] = seconds + micros / 1_000_000
            await client.set(f'vs:resources:{pool}:{worker_id}', json.dumps(sample), ex=60)
        except RedisError:
            logging.getLogger(__name__).warning('worker=%s resource snapshot unavailable', worker_id)
        await asyncio.sleep(SAMPLE_SECONDS)
