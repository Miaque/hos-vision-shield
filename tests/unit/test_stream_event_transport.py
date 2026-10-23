import asyncio
import json
import sys

from vision_shield.media.runner import run_process


async def test_full_stream_sample_window_survives_child_pipe():
    sample = dict(sequence=12345, source_pts='123456789/90000',
                  received_monotonic=4582817.6956143, encoded_monotonic=4582818.6956143,
                  latency_seconds=1.0)
    event = dict(stats=dict(stream=dict(recent_outputs=[sample] * 512)))
    assert len(json.dumps(event)) > 65536
    received = []

    async def collect(value):
        received.append(value)

    result = await run_process([sys.executable, '-u', '-c',
        'import sys; print(sys.stdin.readline().strip(), flush=True)'], event, asyncio.Event(), collect)
    assert result['returncode'] == 0 and received == [event]


async def test_stop_drains_final_stats_before_confirming_exit():
    cancel = asyncio.Event()
    received = []

    async def collect(event):
        received.append(event)
        if event.get('ready'):
            cancel.set()

    command = [sys.executable, '-u', '-c',
        "import sys,json; sys.stdin.readline(); print(json.dumps({'ready':True}),flush=True); "
        "sys.stdin.readline(); print(json.dumps({'stats':{'encoded_frames':12,'padding':'x'*80000}}),flush=True)"]
    result = await run_process(command, {}, cancel, collect, stop_timeout=2)
    assert result['cancelled'] and result['returncode'] == 0
    assert received[-1]['stats']['encoded_frames'] == 12
