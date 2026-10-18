import pytest
import shutil
import subprocess
import time

import httpx

from tests.fixtures.publisher_probe import PublisherProbe, read_headers
from tests.conftest import NO_WINDOW, stop_process

pytestmark = pytest.mark.integration


def test_session_can_be_revoked_after_announce_before_record(tmp_path):
    with PublisherProbe(tmp_path) as probe:
        with probe.announce("g1") as old:
            assert b"200 OK" in read_headers(old)
            assert len(probe.session_ids) == 1
            probe.generation = "g2"
            for session_id in probe.session_ids.copy():
                assert probe.kick(session_id).is_success
            assert old.recv(4096) == b""
        with probe.announce("g1") as rejected:
            assert b"401 Unauthorized" in read_headers(rejected)
        with probe.announce("g2") as current:
            assert b"200 OK" in read_headers(current)


def test_revoking_while_publish_callback_is_still_in_flight(tmp_path):
    with PublisherProbe(tmp_path) as probe:
        probe.release.clear()
        with probe.announce("g1") as old:
            assert probe.approved.wait(5), "MediaMTX did not call the publish check"
            probe.generation = "g2"
            # The callback already approved g1, but ANNOUNCE has not returned.
            # This distinguishes a live session revocation from a path-empty check.
            results = [probe.kick(session_id).status_code for session_id in probe.session_ids.copy()]
            probe.release.set()
            assert results and all(status // 100 == 2 for status in results)
            # ANNOUNCE may have a buffered response; revocation must close the
            # connection before the old publisher can progress to SETUP/RECORD.
            read_headers(old)
            assert old.recv(4096) == b""
        with probe.announce("g1") as rejected:
            assert b"401 Unauthorized" in read_headers(rejected)


def wait_ready(probe):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = httpx.get(f"http://127.0.0.1:{probe.api_port}/v3/paths/get/redacted/probe")
        if response.is_success and response.json().get("ready"):
            return
        time.sleep(0.1)
    raise AssertionError("RTSP publisher did not become ready")


@pytest.mark.parametrize("attempt", range(3))
def test_live_publisher_revoked_after_control_lease_expiry(tmp_path, redis_url, attempt):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "FFmpeg required for live publisher validation"
    with PublisherProbe(tmp_path, redis_url) as probe:
        def start(generation):
            return subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi",
                 "-i", "testsrc2=size=320x240:rate=12", "-c:v", "libx264", "-preset", "ultrafast",
                 "-tune", "zerolatency", "-an", "-rtsp_transport", "tcp", "-f", "rtsp",
                 f"rtsp://{generation}:probe@127.0.0.1:{probe.rtsp_port}/redacted/probe"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
            )

        old = start("g1")
        try:
            wait_ready(probe)
            old_sessions = probe.session_ids.copy()
            # Stop renewal while the independent media process keeps its RTSP socket.
            probe.control.pexpire("probe:generation", 100)
            time.sleep(0.2)
            assert probe.generation is None and old.poll() is None
            for session_id in old_sessions:
                assert probe.kick(session_id).is_success
            old.wait(timeout=10)
            probe.generation = "g2"
            with probe.announce("g1") as rejected:
                assert b"401 Unauthorized" in read_headers(rejected)
            current = start("g2")
            try:
                wait_ready(probe)
                probe.generation = None
                for session_id in probe.session_ids - old_sessions:
                    assert probe.kick(session_id).is_success
                current.wait(timeout=10)
                with probe.announce("g2") as stopped:
                    assert b"401 Unauthorized" in read_headers(stopped)
            finally:
                stop_process(current)
        finally:
            stop_process(old)
