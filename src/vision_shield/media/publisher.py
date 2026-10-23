"""Revoke media-side sessions; a Redis lease alone cannot fence a live socket."""

import asyncio

import httpx

class Publisher:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    async def _change(self, task_id, operation, **arguments):
        """媒体侧也走统一状态转换，冻结和停流不能遗漏通知。"""
        return await self.store.publisher_change(task_id, operation, **arguments)

    async def close(self, task: dict, owner_id: str | None = None, generation: int | None = None) -> bool:
        frozen = await self._change(task["id"], "freeze", owner_id=owner_id, generation=generation)
        if not frozen:
            return False
        sessions = list(frozen["sessions"])
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            for session_id in sessions:
                base = self.settings.mediamtx_control_url
                response = await client.post(f"{base}/v3/rtspsessions/kick/{session_id}")
                if response.status_code not in {200, 404}:
                    response.raise_for_status()
                # Every approved callback registered an already allocated MediaMTX
                # session before returning 204; wait for that exact session to end.
                async with asyncio.timeout(5):
                    while True:
                        current = await client.get(f"{base}/v3/rtspsessions/get/{session_id}")
                        if current.status_code == 404:
                            break
                        current.raise_for_status()
                        await asyncio.sleep(0.05)
        await self._change(task["id"], "clear", owner_id=owner_id, generation=generation, sessions=sessions)
        return True

    async def prepare(self, task: dict, owner_id: str, generation: int) -> str:
        if not await self.close(task, owner_id, generation):
            raise RuntimeError("Media ownership was lost")
        if not await self._change(task["id"], "open", owner_id=owner_id, generation=generation):
            raise RuntimeError("Media ownership was lost")
        base = self.settings.rtsp_publish_base_url or self.settings.rtsp_public_base_url
        url = httpx.URL(f"{base}/redacted/{task['id']}")
        return str(url.copy_with(username=f"{task['id']}.{generation}", password=owner_id))

    async def authorize(self, payload: dict) -> bool:
        try:
            task_id, generation = payload["user"].rsplit(".", 1)
            if payload["action"] != "publish" or payload["path"] != f"redacted/{task_id}":
                return False
            return bool(await self._change(task_id, "authorize", owner_id=payload["password"],
                                            generation=int(generation), session_id=payload["id"]))
        except (KeyError, ValueError):
            return False
