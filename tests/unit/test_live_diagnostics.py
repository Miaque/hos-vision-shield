"""诊断有界、与原计时兼容，不能将发布等待算作模型处理。"""
from types import SimpleNamespace

import pytest

from vision_shield.detection.timing import StageTimes


def test_slow_scopes_keep_their_own_clock_and_preserve_cumulative_timing(monkeypatch):
    from vision_shield.detection import timing
    clock = iter([10., 10.6])
    monkeypatch.setattr(timing, 'perf_counter', lambda: next(clock))
    times = StageTimes()
    observed = []
    times.observer = lambda name, elapsed: observed.append((name, elapsed))
    with times.measure('local-confirm'):
        pass
    assert observed == [('local-confirm', pytest.approx(.6))]
    assert times.snapshot()['local-confirm'] == pytest.approx(.6)


def test_live_diagnostics_separates_frame_processing_from_release_and_is_bounded(monkeypatch):
    from vision_shield.media import live_diagnostics
    detector = SimpleNamespace(stage_times=StageTimes())
    trace = live_diagnostics.LiveDiagnostics(detector, {}, 'attempt', runtime={'source_sha256': {'x': 'a'}})
    clock = iter([10., 10.2, 11., 11.15])
    monkeypatch.setattr(live_diagnostics.time, 'monotonic', lambda: next(clock))
    with trace.operation('process', '123/25'):
        detector.stage_times.seconds['local-confirm'] += .15
    with trace.operation('lookahead-expire', None):
        pass
    data = trace.snapshot()
    assert data['operations'][0]['seconds'] == pytest.approx(.2)
    assert data['operations'][0]['stages']['local-confirm'] == pytest.approx(.15)
    assert data['operations'][0]['source_pts'] == '123/25'
    assert data['operations'][1]['kind'] == 'lookahead-expire'
    assert data['runtime']['source_sha256'] == {'x': 'a'}
    monkeypatch.setattr(live_diagnostics.time, 'monotonic', lambda: 20.)
    for _ in range(200):
        with trace.operation('process', '2'):
            pass
    data = trace.snapshot()
    assert len(data['operations']) == 128 and data['operations_discarded'] == 74
    trace.close()
    assert detector.stage_times.observer is None


def test_slow_scope_samples_do_not_claim_the_trigger_frame_is_the_reviewed_frame(monkeypatch):
    from vision_shield.media.live_diagnostics import LiveDiagnostics
    detector = SimpleNamespace(stage_times=StageTimes())
    trace = LiveDiagnostics(detector, {}, 'a', runtime={})
    for _ in range(100):
        trace.observe('source-backpressure', .2)
    data = trace.snapshot()
    assert len(data['slow_scopes']) == 64
    assert data['slow_scopes_discarded'] == 36
    assert data['slow_scopes'][-1]['seconds'] == .2
    assert 'source_pts' not in data['slow_scopes'][-1]
    trace.close()


def test_delayed_render_keeps_the_original_frame_and_does_not_suppress_a_processing_error():
    from vision_shield.media.live_diagnostics import LiveDiagnostics
    trace = LiveDiagnostics(SimpleNamespace(stage_times=StageTimes()), {}, 'a', runtime={})
    with trace.operation('process', '1'):
        pass
    with pytest.raises(ValueError, match='unsafe'):
        with trace.operation('process', '2'):
            raise ValueError('unsafe')
    with trace.operation('render', '1', frame_sequence=1):
        pass
    data = trace.snapshot()
    assert [(x['kind'], x['frame_sequence']) for x in data['operations']] == [
        ('process', 1), ('process', 2), ('render', 1)]
    assert len(__import__('json').dumps(data).encode()) < 192 * 1024
    trace.close()


async def test_stream_diagnostics_returns_runtime_evidence_with_attempts():
    from unittest.mock import AsyncMock
    import httpx
    from uuid import UUID
    from vision_shield.config import Settings
    from vision_shield.main import create_app
    app = create_app(Settings(_env_file=None, debug=True, stream_diagnostics=False))
    app.state.settings = Settings(_env_file=None, debug=True, stream_diagnostics=False)
    app.state.store = SimpleNamespace(get=AsyncMock(return_value=dict(kind='stream', stream_attempts=[{'run_id': 'a'}],
        device_profile={'mode': 'live', 'attempt_id': 'new', 'operations': []})))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get(f'/v1/tasks/{UUID(int=1)}/diagnostics')
    assert response.status_code == 200
    assert response.json()['live']['attempt_id'] == 'new'


async def test_production_diagnostics_does_not_read_tasks_or_start_receivers():
    from unittest.mock import AsyncMock
    import httpx
    from uuid import UUID
    from vision_shield.config import Settings
    from vision_shield.main import create_app

    settings = Settings(_env_file=None, debug=False, stream_diagnostics=True)
    app = create_app(settings)
    app.state.settings = settings
    app.state.store = SimpleNamespace(get=AsyncMock())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get(f'/v1/tasks/{UUID(int=1)}/diagnostics')
    assert response.status_code == 404
    app.state.store.get.assert_not_called()
    assert not hasattr(app.state, 'live_receivers')
