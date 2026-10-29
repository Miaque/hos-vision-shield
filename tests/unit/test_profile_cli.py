import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize('debug', [None, 'false', 'true'])
def test_local_profile_cli_requires_debug_before_creating_artifacts(tmp_path, monkeypatch, debug):
    from vision_shield.media import child

    if debug is None:
        monkeypatch.delenv('DEBUG', raising=False)
    else:
        monkeypatch.setenv('DEBUG', debug)
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'video fixture')
    target = tmp_path / 'diagnostic'
    calls = []

    def process(payload, stop):
        calls.append(payload)
        assert payload['config']['debug'] is True
        return 0, True

    monkeypatch.setattr(child, 'process', process)
    monkeypatch.setattr(child, 'emit', child.emit)
    monkeypatch.setattr('sys.argv', ['profile-local-clip.py', str(source), str(target),
                                   '--models', str(tmp_path), '--mode', 'host'])
    script = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/profile-local-clip.py'))
    with pytest.raises(SystemExit) as exit_info:
        script['main']()
    assert exit_info.value.code == (0 if debug == 'true' else 2)
    assert bool(calls) is (debug == 'true')
    assert target.exists() is (debug == 'true')
