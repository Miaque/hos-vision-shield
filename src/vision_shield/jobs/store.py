"""Persistent task intent and atomic ownership transitions in one Redis instance."""

import json
from datetime import datetime, timezone
from uuid import uuid4

from redis.asyncio import Redis

TRANSITION = """
local op, arg = ARGV[1], cjson.decode(ARGV[2])
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local raw = redis.call('HGET', KEYS[1], 'payload')
if op == 'create' then
    local pool = arg.kind == 'stream' and KEYS[3] or KEYS[2]
    if redis.call('SCARD', pool) >= tonumber(ARGV[4]) then return 'capacity' end
    redis.call('HSET', KEYS[1], 'payload', ARGV[2])
    redis.call('SADD', pool, arg.id)
    return ARGV[2]
end
if not raw then return false end
local t = cjson.decode(raw)
local pool = t.kind == 'stream' and KEYS[3] or KEYS[2]
local terminal = t.state == 'succeeded' or t.state == 'failed' or t.state == 'cancelled' or t.state == 'stopped'
local owned = t.owner_id == arg.owner_id and t.generation == arg.generation and t.lease_until > now
if op == 'claim' then
    if terminal or t.desired_state ~= 'run' or t.lease_until > now then return false end
    if t.state ~= 'queued' and t.state ~= 'recovering' then return false end
    t.owner_id = arg.owner_id
    t.generation = t.generation + 1
    t.lease_until = now + tonumber(ARGV[3])
    t.dispatch_pending = false
    t.publish_allowed = false
    t.state = t.kind == 'stream' and 'connecting' or 'processing'
elseif op == 'heartbeat' then
    if not owned or terminal then return false end
    t.lease_until = now + tonumber(ARGV[3])
elseif op == 'finish' then
    if not owned or terminal then return false end
    local valid = (arg.state == 'succeeded' or arg.state == 'failed' or arg.state == 'cancelled') and t.kind ~= 'stream'
    valid = valid or (t.kind == 'stream' and (arg.state == 'failed' or arg.state == 'stopped'))
    if not valid then return false end
    if arg.state == 'succeeded' and t.desired_state ~= 'run' then return false end
    t.state, t.result_ref, t.error = arg.state, arg.result_ref, arg.error
    t.ready = arg.state == 'succeeded'
    t.owner_id, t.lease_until = cjson.null, 0
    t.publish_allowed = false
    redis.call('SREM', pool, t.id)
elseif op == 'stop' then
    local expected = t.kind == 'stream' and 'stopped' or 'cancelled'
    if arg.desired_state ~= expected then return 'invalid' end
    if terminal and t.state ~= expected then return 'invalid' end
    t.desired_state, t.ready, t.publish_allowed = expected, false, false
    if t.state == 'queued' and t.owner_id == cjson.null then
        t.state = expected
        redis.call('SREM', pool, t.id)
    end
elseif op == 'recover' then
    if terminal or t.lease_until == 0 or t.lease_until > now then return false end
    t.owner_id, t.lease_until, t.ready, t.publish_allowed = cjson.null, 0, false, false
    if t.kind == 'stream' then
        t.state, t.dispatch_pending = 'recovering', true
    else
        t.state = t.desired_state == 'cancelled' and 'cancelled' or 'failed'
        t.error = {code='worker_lost', message='Processing worker was lost'}
        redis.call('SREM', pool, t.id)
    end
elseif op == 'update' then
    if not owned or terminal or t.desired_state ~= 'run' then return false end
    for k,v in pairs(arg.values) do t[k] = v end
else
    return redis.error_reply('Unknown task operation')
end
t.updated_at = ARGV[5]
local result = cjson.encode(t)
redis.call('HSET', KEYS[1], 'payload', result)
return result
"""


class CapacityExceeded(Exception):
    pass


class InvalidTransition(Exception):
    pass


class TaskStore:
    """Atomic JSON records; ownership operations return None when their CAS loses."""
    def __init__(self, client: Redis, file_limit=100, stream_limit=100, lease_seconds=20):
        self.client = client
        self.file_limit, self.stream_limit = file_limit, stream_limit
        self.lease_ms = int(lease_seconds * 1000)

    async def _change(self, task_id, operation, arguments, limit=0):
        result = await self.client.eval(
            TRANSITION, 3, f"vs:task:{task_id}", "vs:active:files", "vs:active:streams",
            operation, json.dumps(arguments), self.lease_ms, limit,
            datetime.now(timezone.utc).isoformat(),
        )
        if result in (b"capacity", "capacity"):
            raise CapacityExceeded
        if result in (b"invalid", "invalid"):
            raise InvalidTransition
        return json.loads(result) if result else None

    async def create(self, kind: str, source_kind: str, source_ref: str, task_id: str | None = None,
                     *, redaction_mode: str = "solid", profile_detection: bool = False) -> dict:
        if redaction_mode not in {"solid", "mosaic"}:
            raise ValueError("Unsupported redaction mode")
        if profile_detection and kind != 'video':
            raise ValueError('Detection profiling requires a video task')
        task_id = task_id or str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        task = dict(
            id=task_id, kind=kind, redaction_mode=redaction_mode, state="queued", desired_state="run", ready=False,
            created_at=now, updated_at=now, source_kind=source_kind, source_ref=source_ref,
            result_ref=None, owner_id=None, generation=0, lease_until=0,
            dispatch_pending=True, dispatch_until=0, error=None, publish_allowed=False,
            sessions={}, stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0),
        )
        if profile_detection:
            task['profile_detection'] = True
        return await self._change(task_id, "create", task, self.stream_limit if kind == "stream" else self.file_limit)

    async def get(self, task_id: str) -> dict | None:
        payload = await self.client.hget(f"vs:task:{task_id}", "payload")
        return json.loads(payload) if payload else None

    async def active(self) -> list[dict]:
        ids = await self.client.sunion("vs:active:files", "vs:active:streams")
        return [task for task_id in ids if (task := await self.get(task_id.decode() if isinstance(task_id, bytes) else task_id))]

    async def claim(self, task_id: str, owner_id: str) -> dict | None:
        return await self._change(task_id, "claim", dict(owner_id=owner_id))

    async def heartbeat(self, task_id: str, owner_id: str, generation: int) -> dict | None:
        return await self._change(task_id, "heartbeat", dict(owner_id=owner_id, generation=generation))

    async def finish(self, task_id: str, owner_id: str, generation: int, state: str,
                     result_ref: str | None = None, error: dict | None = None) -> dict | None:
        return await self._change(task_id, "finish", dict(owner_id=owner_id, generation=generation, state=state, result_ref=result_ref, error=error))

    async def request_stop(self, task_id: str, desired_state: str) -> dict | None:
        return await self._change(task_id, "stop", dict(desired_state=desired_state))

    async def recover_expired(self) -> None:
        for task in await self.active():
            await self._change(task["id"], "recover", {})

    async def update_owned(self, task_id: str, owner_id: str, generation: int, **values) -> dict | None:
        return await self._change(task_id, "update", dict(owner_id=owner_id, generation=generation, values=values))
