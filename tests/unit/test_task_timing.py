"""模型计时留在内部事件，失败不产生成功媒体。"""

import io
import json
from types import SimpleNamespace

import pytest

from vision_shield.config import Settings
from vision_shield.media import child
from vision_shield.jobs.tasks import log_model_timing
from vision_shield.api.schemas import Stats


@pytest.mark.parametrize("load_fails", [False, True])
def test_child_model_timing_and_failure_have_no_success(tmp_path, monkeypatch, capsys, load_fails):
    from vision_shield import detection
    payload = dict(kind="image", source="rtsp://private:secret@camera/input", directory=str(tmp_path / "attempt"),
                   config=Settings(_env_file=None).model_dump(mode="json"))
    monkeypatch.setattr(child.sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
    monkeypatch.setattr(child.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None))
    def load(config):
        if load_fails:
            raise RuntimeError("private:secret")
        return object()
    monkeypatch.setattr(detection, "load_detector", load)
    with pytest.raises(SystemExit):
        child.main()
    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    timings = [event for event in events if "model_load_seconds" in event]
    assert bool(timings) is not load_fails
    if timings:
        assert timings[0]["model_load_seconds"] >= 0
    assert not any("result_ref" in event for event in events)
    assert "secret" not in output


def test_model_load_time_is_logged_without_changing_response_stats(caplog):
    import logging
    event = dict(model_load_seconds=1.25, source="rtsp://private:secret@camera")
    with caplog.at_level(logging.INFO):
        log_model_timing(event, "job-id", 3)
    assert "job-id" in caplog.text and "1.250" in caplog.text and "secret" not in caplog.text
    assert set(Stats.model_fields) == {"processed_frames", "dropped_frames", "elapsed_seconds"}
