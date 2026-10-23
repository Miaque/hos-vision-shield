"""Async supervision; synchronous inference and codecs remain in a child process."""

import asyncio
import contextlib
import json
import os
import subprocess
from collections.abc import Awaitable, Callable


async def kill_owned_process(process):
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        await killer.wait()
    else:
        process.kill()
    await process.wait()


async def run_process(command: list[str], payload: dict, cancel: asyncio.Event,
                      on_event: Callable[[dict], Awaitable[None]], stop_timeout: float = 10,
                      *, on_exit: Callable[[], Awaitable[None]] | None = None) -> dict:
    """Supervise one child; on_exit confirms completed teardown or failed process creation."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError:
        if on_exit is not None:
            await on_exit()
        raise
    stop = asyncio.create_task(cancel.wait())
    read = None
    result = {}
    try:
        process.stdin.write((json.dumps(payload) + "\n").encode())
        await process.stdin.drain()
        while True:
            read = asyncio.create_task(process.stdout.readline())
            done, _ = await asyncio.wait({read, stop}, return_when=asyncio.FIRST_COMPLETED)
            if stop in done:
                read.cancel()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    process.stdin.write(b'{"stop":true}\n')
                    await process.stdin.drain()
                try:
                    await asyncio.wait_for(process.wait(), stop_timeout)
                except TimeoutError:
                    await kill_owned_process(process)
                result["cancelled"] = True
                break
            line = read.result()
            if not line:
                await process.wait()
                break
            event = json.loads(line)
            result.update(event)
            await on_event(event)
        result["returncode"] = process.returncode
        return result
    finally:
        stop.cancel()
        if read is not None:
            read.cancel()
        await asyncio.gather(stop, *([read] if read else []), return_exceptions=True)
        if process.returncode is None:
            await kill_owned_process(process)
        process.stdin.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()
        if on_exit is not None:
            await on_exit()
