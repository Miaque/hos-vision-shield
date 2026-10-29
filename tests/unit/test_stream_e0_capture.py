"""The experiment client must use GET only and refuse pre-E0 diagnostics."""
import importlib
import json
from pathlib import Path
import sys

import httpx
import pytest

from vision_shield.media.stream_e0 import StreamE0
from vision_shield.media.stream_e0 import runtime_fingerprint


@pytest.mark.parametrize('enabled', [False, True])
def test_capture_client_is_get_only_and_requires_e0(tmp_path, monkeypatch, enabled):
    folder = Path(__file__).resolve().parents[2]/'specs/006-stream-backlog-pacing/experiments'
    monkeypatch.syspath_prepend(str(folder))
    capture = importlib.import_module('e0_capture')
    requests = []
    def respond(request):
        requests.append((request.method, request.url.path))
        if request.url.path == '/v1/tasks/job':
            return httpx.Response(200, json={'state': 'running', 'ready': True})
        live = dict(attempt_id='a', runtime={})
        if enabled:
            live['e0'] = StreamE0().snapshot()
        return httpx.Response(200, json=dict(live=live, attempts=[]))
    client_type = httpx.Client
    monkeypatch.setattr(capture.httpx, 'Client', lambda **kwargs:
                        client_type(**kwargs, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(capture, 'verify_runtime', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['e0_capture.py', str(tmp_path), '--ref', 'candidate', '--task', 'job', '--seconds', '1'])
    if enabled:
        capture.main()
        rows = [json.loads(line) for line in (tmp_path/'e0.jsonl').read_text().splitlines()]
        assert {row['kind'] for row in rows} == {'runtime', 'e0_state'}
    else:
        with pytest.raises(KeyError, match='e0'):
            capture.main()
    assert requests == [('GET', '/v1/tasks/job'), ('GET', '/v1/tasks/job/diagnostics')]


def test_analysis_uses_counters_when_pacer_ring_has_no_retained_records(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[2]/'specs/006-stream-backlog-pacing/experiments'
    monkeypatch.syspath_prepend(str(folder))
    analyze = importlib.import_module('e0_analyze')
    monkeypatch.setattr(analyze, 'verify_runtime', lambda *args: None)
    state = dict(runtime_fingerprint=runtime_fingerprint({}), started_monotonic=0,
        sampled_monotonic=10, ended_monotonic=10, main_wall_seconds=10, main_unattributed_seconds=1,
        main_exclusive_seconds={'wait:pacer': 3, 'op:output-pacing': 2, 'stage:track-recheck': 4},
        pacer_totals={'decisions': 2000, 'actual_wait_seconds': 3,
            'retained_media_wait_seconds': 2, 'conditional_rebase_wait_seconds': 1},
        fifo_totals={}, clock_totals={'video_mux_packets': 2000})
    path = tmp_path/'e0.jsonl'
    initial = dict(state, sampled_monotonic=0, ended_monotonic=None, main_wall_seconds=0,
                   main_unattributed_seconds=0, main_exclusive_seconds={}, pacer_totals={}, clock_totals={})
    rows = [dict(kind='runtime', attempt_id='a', data={}),
            dict(kind='e0_state', attempt_id='a', data=initial),
            dict(kind='e0_state', attempt_id='a', data=state)]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    analyze.analyze(path, 'candidate')
    result = json.loads(path.with_name('e0-summary.json').read_text())['attempts']['a']
    assert result['pacer']['decisions'] == result['pacer_records_missing'] == 2000
    assert result['pacer']['conditional_rebase_wait_seconds'] == 1
    assert sum(result['categories_seconds'].values()) == 10


def test_worker_digest_survives_integral_float_settings_in_redis_json(monkeypatch):
    folder = Path(__file__).resolve().parents[2]/'specs/006-stream-backlog-pacing/experiments'
    monkeypatch.syspath_prepend(str(folder))
    analyze = importlib.import_module('e0_analyze')
    worker = dict(source_sha256={'media.streams': 'fixed'},
                  effective_config=dict(audio_buffer_seconds=16., rtsp_connect_timeout=10.,
                                        rtsp_read_timeout=10., frame_buffer_size=2, audio_enabled=True),
                  output_mux_options={'max_interleave_delta': '100000'})
    transported = dict(worker, effective_config={key: int(value) if type(value) is float else value
                                               for key, value in worker['effective_config'].items()})
    assert runtime_fingerprint(worker) != runtime_fingerprint(transported)
    assert analyze.transported_fingerprint(transported) == runtime_fingerprint(worker)
