"""Revoke media-side sessions; a Redis lease alone cannot fence a live socket."""

import asyncio
import json

import httpx

CONTROL = """
local raw = redis.call('HGET',KEYS[1],'payload')
if not raw then return false end
local t, arg = cjson.decode(raw), cjson.decode(ARGV[2])
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local owner = t.owner_id == arg.owner_id and t.generation == arg.generation
local op = ARGV[1]
if op == 'authorize' then
    if not owner or not t.publish_allowed or t.desired_state ~= 'run' or t.lease_until <= now then return false end
    t.sessions[arg.session_id] = t.generation
elseif op == 'freeze' then
    if not owner and t.desired_state ~= 'stopped' then return false end
    t.publish_allowed, t.ready = false, false
elseif op == 'clear' then
    if not owner and t.desired_state ~= 'stopped' then return false end
    for _,id in ipairs(arg.sessions) do t.sessions[id] = nil end
    if t.desired_state == 'stopped' and t.lease_until <= now and next(t.sessions) == nil then
        t.state, t.owner_id, t.lease_until = 'stopped', cjson.null, 0
        redis.call('SREM',KEYS[2],t.id)
    end
elseif op == 'open' then
    if not owner or t.desired_state ~= 'run' or t.lease_until <= now or next(t.sessions) ~= nil then return false end
    t.publish_allowed = true
else return false end
redis.call('HSET',KEYS[1],'payload',cjson.encode(t))
return cjson.encode(t)
"""


class Publisher:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    async def _change(self, task_id, operation, **arguments):
        result = await self.store.client.eval(CONTROL, 2, f"vs:task:{task_id}", "vs:active:streams",
                                              operation, json.dumps(arguments))
        return json.loads(result) if result else None

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
