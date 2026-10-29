import math

import pytest
from pydantic import ValidationError

from vision_shield.config import Settings


@pytest.mark.parametrize(
    "values",
    [
        {"debug": "invalid"},
        {"stream_output_pacing": "invalid"},
        {"file_task_limit": 0},
        {"stream_task_limit": -1},
        {"message_limit": 0},
        {"frame_buffer_size": 0},
        {"http_connect_timeout": 0},
        {"http_read_timeout": math.inf},
        {"download_timeout": math.nan},
        {"lease_seconds": 5, "heartbeat_seconds": 5},
        {"redis_url": "https://localhost"},
        {"rtsp_public_base_url": "http://localhost:8554"},
        {"source_allowlist": ["0.0.0.0/0"]},
        {"source_allowlist": ["*.example.com"]},
        {"source_allowlist": ["http://camera.local"]},
    ],
)
def test_invalid_settings_fail_at_startup(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_defaults_are_bounded_and_sources_are_denied():
    settings = Settings(_env_file=None)
    assert settings.debug is False
    assert settings.file_task_limit == settings.stream_task_limit == 100
    assert settings.frame_buffer_size == 2
    assert settings.stream_output_pacing is True
    assert settings.source_allowlist == []
    assert settings.heartbeat_seconds < settings.lease_seconds


def test_allowlist_accepts_exact_hosts_addresses_and_networks():
    settings = Settings(
        _env_file=None,
        source_allowlist=["camera.local", "10.10.0.0/24", "192.168.1.15", "::1"],
    )
    assert len(settings.source_allowlist) == 4


def test_environment_values_are_loaded_without_exposing_redis_secret(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("STREAM_OUTPUT_PACING", "false")
    monkeypatch.setenv("FILE_TASK_LIMIT", "3")
    monkeypatch.setenv("SOURCE_ALLOWLIST", '["camera.local"]')
    monkeypatch.setenv("REDIS_URL", "redis://:private-password@localhost:6379/0")
    settings = Settings(_env_file=None)
    assert settings.debug is True
    assert settings.stream_output_pacing is False
    assert settings.file_task_limit == 3
    assert settings.source_allowlist == ["camera.local"]
    assert "private-password" not in repr(settings)
