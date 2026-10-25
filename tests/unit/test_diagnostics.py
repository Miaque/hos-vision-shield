"""资源监测属于 worker 容器；缺失、首样本和陈旧值不能伪装成零。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from vision_shield.config import Settings
from vision_shield.main import create_app


def test_cgroup_cpu_uses_elapsed_window_and_core_quota(tmp_path, monkeypatch):
    from vision_shield.jobs.diagnostics import ResourceSampler
    (tmp_path / 'cpu.stat').write_text('usage_usec 2000000\n')
    (tmp_path / 'cpu.max').write_text('200000 100000\n')
    (tmp_path / 'cpuset.cpus.effective').write_text('0-3,6\n')
    (tmp_path / 'memory.current').write_text('1048576\n')
    (tmp_path / 'memory.max').write_text('max\n')
    sampler = ResourceSampler('cpu', tmp_path)
    monkeypatch.setattr('vision_shield.jobs.diagnostics.time.monotonic', lambda: 10)
    first = sampler.sample()
    assert first['cpu']['percent'] is None
    assert first['cpu']['limit_cores'] == 2
    assert first['memory']['used_bytes'] == 1048576
    assert first['memory']['limit_bytes'] is None
    (tmp_path / 'cpu.stat').write_text('usage_usec 5000000\n')
    monkeypatch.setattr('vision_shield.jobs.diagnostics.time.monotonic', lambda: 12)
    second = sampler.sample()
    assert second['sample_window_seconds'] == 2
    assert second['cpu']['percent'] == 150
    assert second['gpu']['devices'] == []
    assert second['gpu']['unavailable_reason'] == 'cpu_inference'


def test_missing_cgroup_is_unavailable_and_not_host_cpu(tmp_path):
    from vision_shield.jobs.diagnostics import ResourceSampler
    sample = ResourceSampler('cpu', tmp_path).sample()
    assert sample['cpu']['percent'] is None
    assert sample['cpu']['unavailable_reason'] == 'cgroup_v2_unavailable'
    assert sample['memory']['used_bytes'] is None


def test_gpu_queries_only_configured_device_and_identifies_shared_device_scope(monkeypatch, tmp_path):
    from vision_shield.jobs import diagnostics
    seen = []
    monkeypatch.setenv('GPU_DEVICE_ID', 'GPU-expected')
    def run(command, **kwargs):
        seen.append(command)
        assert kwargs['timeout'] <= 2 and kwargs.get('shell', False) is False
        return SimpleNamespace(stdout='GPU-expected, NVIDIA RTX 6000 Ada Generation, 74, 1676, 49140, 62.5\n')
    monkeypatch.setattr(diagnostics.subprocess, 'run', run)
    sample = diagnostics.ResourceSampler('cuda', tmp_path).sample()
    assert '--id=GPU-expected' in seen[0]
    assert sample['gpu']['scope'] == 'device'
    assert sample['gpu']['devices'][0]['utilization_percent'] == 74
    assert sample['gpu']['devices'][0]['memory_used_bytes'] == 1676 * 1024**2
    assert sample['gpu']['unavailable_reason'] is None


@pytest.mark.parametrize('stdout', ['GPU-a, RTX, [N/A], [N/A], 49140, [N/A]\n', 'invalid'])
def test_unsupported_gpu_counters_are_null_and_bad_output_is_unavailable(monkeypatch, tmp_path, stdout):
    from vision_shield.jobs import diagnostics
    monkeypatch.setenv('GPU_DEVICE_ID', '0')
    monkeypatch.setattr(diagnostics.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=stdout))
    gpu = diagnostics.ResourceSampler('cuda', tmp_path).sample()['gpu']
    if stdout == 'invalid':
        assert gpu['devices'] == [] and gpu['unavailable_reason'] == 'gpu_query_failed'
    else:
        assert gpu['devices'][0]['utilization_percent'] is None
        assert gpu['devices'][0]['memory_used_bytes'] is None


async def test_resource_api_returns_worker_samples_with_age_and_stale_status():
    from vision_shield.jobs.diagnostics import ResourceSampler
    sample = ResourceSampler('cpu').sample()
    sample['sampled_at_unix'] = 93
    redis = SimpleNamespace(time=AsyncMock(return_value=(100, 0)),
                            zrangebyscore=AsyncMock(side_effect=[[b'worker-1'], [b'worker-2']]),
                            mget=AsyncMock(return_value=[json.dumps(sample), None]))
    app = create_app(Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None, debug=True)
    app.state.redis = redis
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        response = await client.get('/v1/diagnostics/resources')
    assert response.status_code == 200
    assert response.json()['sampled_at_unix'] == 100
    workers = response.json()['workers']
    assert workers[0]['snapshot']['sampled_at_unix'] == 93
    assert workers[0]['pool'] == 'files' and workers[0]['age_seconds'] == 7
    assert workers[0]['stale'] is True
    assert workers[1]['snapshot'] is None and workers[1]['unavailable_reason'] == 'no_sample'
    assert redis.zrangebyscore.call_args_list[0].args == ('vs:workers:files', 80000, '+inf')


@pytest.mark.parametrize('workers,samples,reason', [([], [], None), ([b'worker'], [b'invalid'], 'invalid_sample')])
async def test_resource_api_handles_empty_pools_and_bad_samples(workers, samples, reason):
    redis = SimpleNamespace(time=AsyncMock(return_value=(100, 0)),
                            zrangebyscore=AsyncMock(side_effect=[workers, []]),
                            mget=AsyncMock(return_value=samples))
    app = create_app(Settings(_env_file=None))
    app.state.settings = Settings(_env_file=None, debug=True)
    app.state.redis = redis
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        response = await client.get('/v1/diagnostics/resources')
    assert response.status_code == 200
    if reason:
        assert response.json()['workers'][0]['unavailable_reason'] == reason
        assert response.json()['workers'][0]['stale'] is True
    else:
        assert response.json()['workers'] == []
        redis.mget.assert_not_called()


def test_gpu_unavailable_reasons_and_missing_selection(monkeypatch, tmp_path):
    import subprocess
    from vision_shield.jobs import diagnostics
    monkeypatch.delenv('GPU_DEVICE_ID', raising=False)
    sampler = diagnostics.ResourceSampler('cuda', tmp_path)
    assert sampler.sample()['gpu']['unavailable_reason'] == 'gpu_selection_unavailable'
    monkeypatch.setenv('GPU_DEVICE_ID', '0')
    for error, reason in [(FileNotFoundError(), 'nvidia_smi_unavailable'),
                          (subprocess.TimeoutExpired('nvidia-smi', 1), 'gpu_query_timeout')]:
        def fails(*args, **kwargs):
            raise error
        monkeypatch.setattr(diagnostics.subprocess, 'run', fails)
        assert sampler.sample()['gpu']['unavailable_reason'] == reason


async def test_resource_publisher_uses_expiring_snapshots_and_survives_redis_error(monkeypatch):
    import asyncio
    from redis.exceptions import ConnectionError
    from vision_shield.jobs import diagnostics
    client = SimpleNamespace(time=AsyncMock(side_effect=[ConnectionError(), (100, 0)]), set=AsyncMock())
    async def sample_sleep(seconds):
        if client.set.called:
            raise asyncio.CancelledError
    monkeypatch.setattr(diagnostics.asyncio, 'sleep', sample_sleep)
    with pytest.raises(asyncio.CancelledError):
        await diagnostics.observe_resources(client, 'files', 'worker', 'cpu')
    args, kwargs = client.set.call_args
    assert args[0] == 'vs:resources:files:worker' and kwargs['ex'] == 60
    assert json.loads(args[1])['sampled_at_unix'] == 100
