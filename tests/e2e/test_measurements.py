"""Single-host measurements, including a visible clock carried through RTSP pixels."""

import asyncio
import json
import os
import platform
import subprocess
import threading
import time

import av
import numpy as np

from tests.conftest import NO_WINDOW, ROOT, stop_process
from vision_shield.config import Settings


def resource_snapshot(processes):
    roots = ",".join(str(process.pid) for process in processes)
    script = f"""
    $ids=@({roots}); $all=@(Get-CimInstance Win32_Process)
    do {{
      $children=@($all | Where-Object {{ $_.ParentProcessId -in $ids -and $_.ProcessId -notin $ids }})
      $ids+=@($children | ForEach-Object {{ $_.ProcessId }})
    }} while ($children.Count -gt 0)
    @($all | Where-Object {{ $_.ProcessId -in $ids }} | ForEach-Object {{
      @{{pid=$_.ProcessId; cpu=([double]$_.KernelModeTime+[double]$_.UserModeTime)/1e7; bytes=[double]$_.WorkingSetSize}}
    }}) | ConvertTo-Json -Compress
    """
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True,
                            capture_output=True, text=True, creationflags=NO_WINDOW, timeout=15)
    rows = json.loads(result.stdout)
    return rows if isinstance(rows, list) else [rows]


def collect_stream(url):
    latency, video_times, audio_times = [], [], []
    with av.open(url, options={"rtsp_transport": "tcp"}, timeout=(10, 10)) as reader:
        assert len(reader.streams.audio) == 1
        for packet in reader.demux():
            if packet.stream.type == "audio" and packet.pts is not None:
                audio_times.append(float(packet.pts * packet.time_base))
            elif packet.stream.type == "video":
                for frame in packet.decode():
                    pixels = frame.to_ndarray(format="rgb24")
                    bits = "".join("1" if pixels[8, index * 8 + 4].mean() > 128 else "0" for index in range(40))
                    assert int(bits[:8], 2) == 0xA5, "Visible source clock was not preserved"
                    delay = ((int(time.time() * 1000) & 0xFFFFFFFF) - int(bits[8:], 2)) & 0xFFFFFFFF
                    assert delay < 60000
                    latency.append(delay)
                    if frame.pts is not None:
                        video_times.append(float(frame.pts * frame.time_base))
                    if len(latency) == 5:
                        assert audio_times
                        return dict(latency_ms=latency,
                                    nearest_audio_packet_ms=[1000 * min(abs(t - a) for a in audio_times) for t in video_times])
    raise AssertionError(f"Output stream ended after {len(latency)} frames; five are required")


async def test_reproducible_native_measurements(native_app, generated_media):
    assert os.name == "nt", "This resource measurement entry uses Windows process counters"
    client, snapshots, query_ms = native_app["client"], [], []
    artifacts = ROOT / "data/validation"
    artifacts.mkdir(parents=True, exist_ok=True)
    stopped_sampling = asyncio.Event()

    async def sample_resources():
        while not stopped_sampling.is_set():
            snapshots.append(dict(time=time.time(), rows=await asyncio.to_thread(resource_snapshot, native_app["processes"])))
            try:
                await asyncio.wait_for(stopped_sampling.wait(), 1)
            except TimeoutError:
                pass

    async def poll(task_id, states):
        async with asyncio.timeout(120):
            while True:
                started = time.perf_counter()
                task = (await client.get(f"/v1/tasks/{task_id}")).json()
                query_ms.append(1000 * (time.perf_counter() - started))
                assert task["state"] != "failed", task
                if task["state"] in states:
                    return task
                await asyncio.sleep(0.2)

    sampler = asyncio.create_task(sample_resources())
    report = dict(date=time.strftime("%Y-%m-%d"), platform=platform.platform(), python=platform.python_version(),
                  logical_cpus=os.cpu_count(), concurrency=1,
                  detector_backend=Settings().detection_backend,
                  detector_longest_side=Settings().detection_max_side, inference_threads=2, complete=False)
    source, producer, stop_source = None, None, threading.Event()
    try:
        for kind, source_path, name in [("image", ROOT / "samples/official-demo.jpg", "official-demo-redacted.jpg"),
                                        ("video", generated_media / "aac.mp4", "video-redacted.mp4")]:
            started = time.perf_counter()
            with source_path.open("rb") as media:
                response = await client.post("/v1/file-tasks/uploads", data={"kind": kind}, files={"file": (source_path.name, media)})
            assert response.status_code == 202, response.text
            task = await poll(response.json()["id"], {"succeeded"})
            report[kind] = dict(total_seconds=time.perf_counter() - started, stats=task["stats"])
            (artifacts / name).write_bytes((await client.get(task["result_url"])).content)
        source_url = f"rtsp://127.0.0.1:{native_app['rtsp_port']}/test-source"
        source = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "320x240",
             "-framerate", "4", "-i", "pipe:0", "-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
             "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "4", "-c:a", "aac",
             "-f", "rtsp", "-rtsp_transport", "tcp", source_url],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)

        def produce():
            try:
                while not stop_source.is_set():
                    pixels = np.full((240, 320, 3), 120, np.uint8)
                    stamp = f"{0xA5:08b}{int(time.time() * 1000) & 0xFFFFFFFF:032b}"
                    for index, bit in enumerate(stamp):
                        pixels[:16, index * 8:(index + 1) * 8] = 220 if bit == "1" else 20
                    source.stdin.write(pixels.tobytes())
                    source.stdin.flush()
                    stop_source.wait(0.25)
            except (OSError, ValueError):
                pass

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        await asyncio.sleep(1)
        response = await client.post("/v1/stream-tasks", json={"url": source_url})
        assert response.status_code == 202
        task_id = response.json()["id"]
        running = await poll(task_id, {"running"})
        report["stream"] = await asyncio.wait_for(asyncio.to_thread(collect_stream, running["playback_url"]), 30)
        report["stream"]["stats"] = (await client.get(f"/v1/tasks/{task_id}")).json()["stats"]
        started = time.perf_counter()
        assert (await client.patch(f"/v1/tasks/{task_id}", json={"desired_state": "stopped"})).status_code == 202
        report["stop_ack_ms"] = 1000 * (time.perf_counter() - started)
        await poll(task_id, {"stopped"})
        report["stop_complete_ms"] = 1000 * (time.perf_counter() - started)
        report["complete"] = True
    finally:
        stop_source.set()
        if source:
            stop_process(source)
            source.stdin.close()
        if producer:
            producer.join(timeout=3)
        stopped_sampling.set()
        await sampler
        report.update(query_ms=query_ms, resources=snapshots)
        (artifacts / "performance.json").write_text(json.dumps(report, indent=2), "utf-8")
    assert snapshots and query_ms
