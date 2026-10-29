"""Real CPU codec equality plus the existing CUDA-path contract harness."""
import av
import pytest

from tests.unit.test_stream_audio_clock import test_live_audio_clock_recovery_preserves_audio_and_sync as run_audio
from tests.unit.test_stream_cuda_path import test_cuda_stream_keeps_device_path_order_clocks_and_fails_closed as run_cuda_contract


@pytest.mark.parametrize('protocol,codec,fault,bframes', [
    ('rtsp', 'aac', 'none', 0), ('rtmp', 'aac', 'none', 0),
    ('rtsp', 'aac', 'both_reset', 2), ('rtmp', 'aac', 'audio_repeat', 0),
    ('rtmp', 'pcm_alaw', 'both_reset', 0),
])
def test_e0_keeps_encoded_video_audio_bytes_clocks_and_decoded_masks(tmp_path, monkeypatch, protocol, codec, fault, bframes):
    outputs = []
    for enabled in (False, True):
        root = tmp_path / str(enabled)
        root.mkdir()
        captured = {}
        with monkeypatch.context() as scoped:
            run_audio(root, scoped, protocol, codec, fault, source_bframes=bframes,
                      e0=enabled, capture=captured)
        with av.open(str(captured['target'])) as reader:
            outputs.append([(p.stream.type, bytes(p), p.pts, p.dts, str(p.time_base))
                            for p in reader.demux() if p.size])
        live = captured['events'][-2].get('device_profile')
        assert live is not None
        assert ('e0' in live) is enabled
        if enabled:
            trace = live['e0']
            assert trace['totals']['frames'] == 120
            assert trace['fifo_totals']['enqueued'] == trace['fifo_totals']['dequeued'] == 120
            assert trace['clock_totals']['audio_packets'] > 0
            assert trace['events'][-1]['name'] == 'reader_join_completed'
            assert trace['first_output']['monotonic'] <= trace['last_output']['monotonic']
            assert captured['events'][-1]['stream_end']['e0']['fifo_totals'] == trace['fifo_totals']
    assert outputs[0] == outputs[1]


def test_e0_cuda_contract_retains_every_frame_clock_lookahead_and_pacing_decision(monkeypatch):
    outputs = []
    for enabled in (False, True):
        captured = {}
        with monkeypatch.context() as scoped:
            run_cuda_contract(scoped, None, False, False, True, e0=enabled, capture=captured)
        outputs.append((captured['masked'], captured['encoded']))
        live = captured['events'][-2]['device_profile']
        assert ('e0' in live) is enabled
        if enabled:
            trace = live['e0']
            assert trace['pacer_totals']['decisions'] == 3
            assert captured['events'][-1]['stream_end']['e0']['pacer_totals'] == trace['pacer_totals']
            assert [f['frame_id'] for f in trace['pacer']] == [1, 2, 3]
            assert [f['encoder_packet_clock']['pts'] for f in trace['pacer']] == [0, 3600, 7200]
            assert all(row['encoded_vcl_sha256'] and row['mux_completed_monotonic'] >= row['actual']
                       for row in trace['pacer'])
            for row in trace['frames']:
                assert row['enqueue_monotonic'] <= row['dequeue_monotonic']
                assert row['lookahead_entered_monotonic'] <= row['lookahead_released_monotonic']
                assert row['render_started_monotonic'] <= row['render_completed_monotonic']
                assert row['render_completed_monotonic'] <= row['mux_completed_monotonic']
    assert outputs[0] == outputs[1]
