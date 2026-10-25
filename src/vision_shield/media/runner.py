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
                      *, on_exit: Callable[[], Awaitable[None]] | None = None,
                      session=None) -> dict:
    """Confirm media teardown before on_exit; profiling retains a cold, isolated child."""
    if session is not None:
        if not payload.get('profile_detection') or not payload.get('config', {}).get('debug', False):
            return await session.run(payload, cancel, on_event, stop_timeout, on_exit=on_exit)
        await session.close()
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            # A bounded 512-frame timing window exceeds asyncio's default 64 KiB line.
            limit=256 * 1024,
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
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    process.stdin.write(b'{"stop":true}\n')
                    await process.stdin.drain()
                try:
                    # Keep consuming final flush/stats events, including a read already completed.
                    async with asyncio.timeout(stop_timeout):
                        line = await read
                        while line:
                            event = json.loads(line)
                            result.update(event)
                            await on_event(event)
                            line = await process.stdout.readline()
                        await process.wait()
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


class MediaProcess:
    """One sequential media slot; completion acknowledges closed media, not process EOF."""

    def __init__(self, command):
        self.command, self.process = command, None
        self.model_config = None

    async def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        # Drain while reaping so a full or closing stdout pipe cannot retain the transport.
        await asyncio.gather(kill_owned_process(process), process.stdout.read())
        process.stdin.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()

    async def run(self, payload, cancel, on_event, stop_timeout=10, *, on_exit=None):
        reusable = False
        try:
            if cancel.is_set():
                return dict(cancelled=True, returncode=0)
            model_config = {key: payload['config'].get(key) for key in (
                'debug', 'detection_backend', 'inference_device', 'face_model_path', 'plate_model_path',
                'egoblur_face_model_path', 'egoblur_plate_model_path', 'detection_max_side', 'detection_box_padding')}
            if self.process is not None and (self.process.returncode is not None or model_config != self.model_config):
                await self.close()
            if self.process is None:
                self.process = await asyncio.create_subprocess_exec(
                    *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, limit=256 * 1024,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                self.model_config = model_config
                async with asyncio.timeout(120):
                    result = await self._exchange(dict(kind='probe', config=payload['config']), cancel, on_event, stop_timeout)
                if payload['kind'] == 'probe' or result.get('returncode') != 0 or result.get('cancelled'):
                    reusable = result.pop('_complete', False) and result.pop('reusable', False)
                    return result
            result = await self._exchange(payload, cancel, on_event, stop_timeout)
            reusable = result.pop('_complete', False) and result.pop('reusable', False)
            return result
        finally:
            if not reusable:
                await self.close()
            if on_exit is not None:
                await on_exit()

    async def _exchange(self, payload, cancel, on_event, stop_timeout):
        process = self.process
        stop, read = asyncio.create_task(cancel.wait()), None
        result = {}

        async def consume(line):
            if not line:
                await process.wait()
                result['returncode'] = process.returncode
                return True
            event = json.loads(line)
            if event.pop('complete', False):
                result.update(event, _complete=True)
                return True
            result.update(event)
            await on_event(event)
            return False

        try:
            process.stdin.write((json.dumps(payload) + '\n').encode())
            await process.stdin.drain()
            while True:
                read = asyncio.create_task(process.stdout.readline())
                done, _ = await asyncio.wait({read, stop}, return_when=asyncio.FIRST_COMPLETED)
                if stop in done:
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        process.stdin.write(b'{"stop":true}\n')
                        await process.stdin.drain()
                    try:
                        async with asyncio.timeout(stop_timeout):
                            while not await consume(await read):
                                read = asyncio.create_task(process.stdout.readline())
                    except TimeoutError:
                        await self.close()
                        result['returncode'] = process.returncode
                    result['cancelled'] = True
                    return result
                if await consume(read.result()):
                    return result
        finally:
            stop.cancel()
            if read is not None:
                read.cancel()
            await asyncio.gather(stop, *([read] if read else []), return_exceptions=True)
