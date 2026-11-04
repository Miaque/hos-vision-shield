import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('received', [True, False])
def test_isolated_comparison_uses_mosaic_and_retains_evidence_before_completeness_check(tmp_path, monkeypatch, received):
    path = Path(__file__).parents[2] / 'specs/008-my-server-live-stall/isolated-media.py'
    spec = importlib.util.spec_from_file_location('isolated_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'rtsp-peer.py').write_text('''class RTSPCapture:
    def __init__(self, **kwargs):
        self.url, self.sdp, self.reports = 'rtsp://127.0.0.1/internal', 'test', []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def pictures(self): return PICTURES
    def audio(self): return []
'''.replace('PICTURES', "[dict(at=1, timestamp=2, payload=b'picture')]" if received else '[]'))
    observed = []
    monkeypatch.setattr(module, 'process_stream', lambda source, target, detector, config, stop, emit:
                        observed.append((target, config)))
    if received:
        module.run(tmp_path, 'rtsp://source', 1)
    else:
        with pytest.raises(AssertionError, match='No real RTP pictures'):
            module.run(tmp_path, 'rtsp://source', 1)
    assert observed[0][0] == 'rtsp://127.0.0.1/internal'
    assert observed[0][1]['redaction_mode'] == 'mosaic'
    saved = json.loads((tmp_path / 'isolated-no-model.json').read_text())
    assert saved['redaction_mode'] == 'mosaic'
    assert len(saved['pictures']) == int(received)
