import pytest
from types import SimpleNamespace

from vision_shield.detection.device_profile import SCOPE, busy_ms, cpu_operators, summarize


def event(category, name, start, duration, thread=9, correlation=None, **args):
    if correlation is not None:
        args['correlation'] = correlation
    return dict(ph='X', cat=category, name=name, ts=start, dur=duration, tid=thread, args=args)


def test_trace_correlates_only_model_thread_and_excludes_profiler_cleanup():
    trace = dict(traceEvents=[event('user_annotation', SCOPE, 1000, 10000),
        event('cuda_runtime', 'cudaLaunchKernel', 1100, 20, correlation=1),
        event('cuda_runtime', 'cudaMemcpyAsync', 2000, 100, correlation=2),
        event('cuda_runtime', 'cudaStreamSynchronize', 3000, 4000),
        event('cuda_driver', 'cuStreamSynchronize', 3100, 3800),
        event('cuda_runtime', 'cudaMalloc', 8000, 500),
        event('cuda_runtime', 'cudaDeviceSynchronize', 12000, 9000),
        event('cuda_runtime', 'cudaLaunchKernel', 1500, 20, thread=10, correlation=3),
        event('kernel', 'plate-kernel', 1500, 1000, thread=7, correlation=1),
        event('gpu_memcpy', 'Memcpy DtoH', 2200, 500, thread=7, correlation=2, bytes=64),
        event('kernel', 'other-model-kernel', 1500, 8000, thread=8, correlation=3)])
    result = summarize(trace)
    assert result['host_forward_ms'] == 10
    assert result['kernel_count'] == 1 and result['kernel_sum_ms'] == 1
    assert result['gpu_busy_ms'] == 1.2
    assert result['synchronization_api_events'] == 2 and result['synchronization_host_ms'] == 4
    assert result['allocation_host_ms'] == 0.5
    assert result['DtoH'] == dict(calls=1, milliseconds=0.5, bytes=64)
    assert result['HtoD'] == dict(calls=0, milliseconds=0, bytes=0)
    assert result['top_kernels'][0]['name'] == 'plate-kernel'


@pytest.mark.parametrize('events', [[], [event('user_annotation', SCOPE, 0, 10)]])
def test_missing_scope_or_cuda_activity_is_not_a_successful_device_profile(events):
    with pytest.raises(RuntimeError):
        summarize(dict(traceEvents=events))


def test_gpu_busy_time_is_union_not_sum_of_overlapping_kernels():
    assert busy_ms([dict(ts=0, dur=2000), dict(ts=1000, dur=200), dict(ts=1500, dur=1000)]) == 2.5


def test_cpu_operator_table_excludes_other_threads_and_profiler_cleanup():
    def cpu(name, start, end, own, thread=4, device='CPU'):
        return SimpleNamespace(name=name, thread=thread, device_type=SimpleNamespace(name=device),
                               time_range=SimpleNamespace(start=start, end=end),
                               self_cpu_time_total=own, self_device_time_total=0)
    rows = cpu_operators([cpu(SCOPE, 100, 200, 0), cpu('aten::sum', 110, 120, 10),
                          cpu('aten::sum', 130, 150, 20), cpu('cudaDeviceSynchronize', 210, 5000, 4790),
                          cpu('other-model', 120, 150, 30, thread=5), cpu('GPU kernel', 120, 140, 0, device='CUDA')])
    assert rows == [dict(name='aten::sum', name_truncated=False, calls=2, self_cpu_ms=0.03, self_device_ms=0)]


def test_local_calls_and_unsampled_frames_never_start_profiler(tmp_path, monkeypatch):
    import torch.profiler
    from vision_shield.detection.device_profile import CHECKPOINTS, DeviceSamples

    monkeypatch.setattr(torch.profiler, 'profile', lambda **kw: pytest.fail('not a selected global frame'))
    events, calls = [], []
    sampler = DeviceSamples(tmp_path, events.append)
    def model(pixels):
        calls.append(pixels)
        return pixels
    for _ in range(100):
        assert sampler.forward(model, 5, global_view=False) == 5
    for _ in range(CHECKPOINTS[0] - 1):
        assert sampler.forward(model, 7, global_view=True) == 7
    assert sampler.frames == CHECKPOINTS[0] - 1 and sampler.samples == []
    sampler.finish()
    assert events == [dict(device_profile=dict(state='partial', samples=[]))]
    assert len(calls) == 100 + CHECKPOINTS[0] - 1


def test_trace_budget_failure_is_explicit_and_does_not_rerun_model(tmp_path, monkeypatch):
    import torch.profiler
    from vision_shield.detection import device_profile

    class Profiler:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def export_chrome_trace(self, target):
            from pathlib import Path
            Path(target).write_text('{}', 'utf-8')
    monkeypatch.setattr(torch.profiler, 'profile', lambda **kw: Profiler())
    monkeypatch.setattr(device_profile, 'CHECKPOINTS', (1,))
    monkeypatch.setattr(device_profile, 'MAX_TRACE_BYTES', 1)
    events, calls = [], []
    sampler = device_profile.DeviceSamples(tmp_path, events.append)
    def model(pixels):
        calls.append(pixels)
        return pixels
    with pytest.raises(RuntimeError, match='budget'):
        sampler.forward(model, SimpleNamespace(), global_view=True)
    assert len(calls) == 1
    assert events[-1]['device_profile']['state'] == 'failed'
    assert not list(tmp_path.glob('plate-profile-*'))


def test_external_ids_correlate_recreated_threads_without_including_parallel_models():
    trace = dict(traceEvents=[event('user_annotation', SCOPE, 1000, 1000, **{'External id': 10}),
        event('cpu_op', 'aten::model', 1050, 500, **{'External id': 11}),
        event('cpu_op', 'other-model', 1050, 500, thread=7, **{'External id': 12}),
        event('cpu_op', 'outside-scope', 2100, 500, **{'External id': 13}),
        event('cuda_runtime', 'cudaLaunchKernel', 1100, 20, thread=99, correlation=1, **{'External id': 11}),
        event('cuda_runtime', 'cudaMemcpyAsync', 1600, 10, thread=99, correlation=2, **{'External id': 10}),
        event('cuda_runtime', 'cudaLaunchKernel', 1100, 20, thread=99, correlation=3, **{'External id': 12}),
        event('cuda_runtime', 'cudaLaunchKernel', 1100, 20, correlation=4, **{'External id': 13}),
        event('kernel', 'selected-model', 1200, 100, thread=2, correlation=1),
        event('gpu_memcpy', 'Memcpy DtoH', 1700, 50, thread=2, correlation=2, bytes=64),
        event('kernel', 'parallel-model', 1200, 700, thread=3, correlation=3),
        event('kernel', 'outside-parent', 1200, 700, thread=3, correlation=4)])
    result = summarize(trace)
    assert result['kernel_count'] == 1 and result['top_kernels'][0]['name'] == 'selected-model'
    assert result['cuda_api_events'] == 2
    assert result['DtoH'] == dict(calls=1, milliseconds=0.05, bytes=64)
