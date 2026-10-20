import asyncio
import sys
import json
import subprocess
import os

import pytest

from vision_shield.config import Settings
from tests.conftest import ROOT, NO_WINDOW

from vision_shield.media.runner import run_process


async def test_stuck_child_is_killed_without_blocking_event_loop():
    cancel = asyncio.Event()
    ticks = []

    async def on_event(event):
        if event.get("ready"):
            cancel.set()

    async def heartbeat():
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks.append(True)

    command = [sys.executable, "-u", "-c", "import time; print('{\"ready\":true}',flush=True); time.sleep(30)"]
    result, _ = await asyncio.gather(
        run_process(command, {}, cancel, on_event, stop_timeout=0.2), heartbeat()
    )
    assert result["cancelled"] and result["returncode"] is not None
    assert len(ticks) == 5


def test_parent_pipe_eof_terminates_real_media_child(tmp_path):
    payload = dict(kind="image", source=str(ROOT / "samples/official-demo.jpg"), directory=str(tmp_path),
                   config=Settings().model_dump(mode="json"))
    process = subprocess.Popen([sys.executable, "-u", "-m", "vision_shield.media.child"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               creationflags=NO_WINDOW)
    try:
        process.stdin.write((json.dumps(payload) + "\n").encode())
        process.stdin.flush()
        process.stdin.close()
        assert process.wait(timeout=30) == 3
        assert b"result_ref" not in process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="Linux 冷加载父管道保护须在 POSIX 实测")
def test_parent_eof_terminates_during_model_loading(tmp_path):
    command = [sys.executable, "-u", "-c", "import time; import vision_shield.detection as d; "
               "d.load_detector=lambda cfg: time.sleep(30); "
               "from vision_shield.media.child import main; main()"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        process.stdin.write((json.dumps(dict(kind="image", config={}, directory=str(tmp_path))) + "\n").encode())
        process.stdin.close()
        assert process.wait(timeout=5) == 3
        assert b"result_ref" not in process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
