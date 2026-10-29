"""Persistent task intent and atomic ownership transitions in one Redis instance."""

import json
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from redis.asyncio import Redis

TRANSITION = """
local op, arg = ARGV[1], cjson.decode(ARGV[2])
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local raw = redis.call('HGET', KEYS[1], 'payload')
if op == 'create' then
    arg._observed_ms = now
    if arg.submission_key ~= cjson.null then
        local binding = redis.call('GET', KEYS[4])
        if binding then
            local previous = redis.call('HGET', 'vs:task:'..binding, 'payload')
            if not previous then return 'invalid' end
            local replay = cjson.decode(previous)
            if replay.fingerprint ~= arg.fingerprint then return 'conflict' end
            replay._observed_ms = now
            return cjson.encode(replay)
        end
    end
    if raw then return 'conflict' end
    local pool = arg.kind == 'stream' and KEYS[3] or KEYS[2]
    if redis.call('SCARD', pool) >= tonumber(ARGV[4]) then return 'capacity' end
    if arg.notification_target ~= cjson.null then
        if redis.call('SCARD', KEYS[5]) >= tonumber(ARGV[6]) then return 'capacity' end
        redis.call('SADD', KEYS[5], arg.id)
    end
    local created = cjson.encode(arg)
    redis.call('HSET', KEYS[1], 'payload', created)
    redis.call('SADD', pool, arg.id)
    if arg.submission_key ~= cjson.null then redis.call('SET', KEYS[4], arg.id) end
    return created
end
if not raw then return false end
local t = cjson.decode(raw)
local old_state, old_ready, old_desired = t.state, t.ready, t.desired_state
local pool = t.kind == 'stream' and KEYS[3] or KEYS[2]
local terminal = t.state == 'succeeded' or t.state == 'failed' or t.state == 'cancelled' or t.state == 'stopped'
local owned = t.owner_id == arg.owner_id and t.generation == arg.generation and t.lease_until > now
local external = t.publish_url ~= nil and t.publish_url ~= cjson.null
if op == 'claim' then
    if terminal or t.desired_state ~= 'run' or t.lease_until > now then return false end
    if t.state ~= 'queued' and t.state ~= 'recovering' then return false end
    if external and t.generation > 0 and t.publisher_owner_id ~= cjson.null then return false end
    if external then t.publisher_owner_id = cjson.null end
    t.owner_id = arg.owner_id
    t.generation = t.generation + 1
    t.lease_until = now + tonumber(ARGV[3])
    t.dispatch_pending = false
    t.publish_allowed = false
    t.last_output_ms = cjson.null
    t.state = t.kind == 'stream' and 'connecting' or 'processing'
elseif op == 'heartbeat' then
    if not owned or terminal then return false end
    t.lease_until = now + tonumber(ARGV[3])
elseif op == 'finish' then
    if not owned or terminal then return false end
    if external and t.publisher_owner_id ~= cjson.null then return false end
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
    -- 旧版本外部任务缺少退出标记时，保留发布者身份，不能推断已经退出。
    if external and t.publisher_owner_id == nil then t.publisher_owner_id = t.owner_id end
    t.owner_id, t.lease_until, t.ready, t.publish_allowed = cjson.null, 0, false, false
    if t.kind == 'stream' then
        t.state, t.dispatch_pending = 'recovering', true
    else
        t.state = t.desired_state == 'cancelled' and 'cancelled' or 'failed'
        t.error = {code='worker_lost', message='Processing worker was lost'}
        redis.call('SREM', pool, t.id)
    end
elseif op == 'update' then
    if not owned or terminal then return false end
    if t.desired_state ~= 'run' then
        if t.kind ~= 'stream' then return false end
        -- 收尾统计可写入，但停止请求不能被迟到的运行状态覆盖。
        local metrics = {stats=true, timings=true, detection_stats=true, video_encoder=true, video_decoder=true, media_failure=true, stream_attempts=true, stream_attempts_discarded=true}
        for k,_ in pairs(arg.values) do if not metrics[k] then return false end end
    end
    for k,v in pairs(arg.values) do t[k] = v end
elseif op == 'output' then
    if not owned or terminal or t.desired_state ~= 'run' or not t.publish_allowed then return false end
    if arg.age_ms < 0 or arg.age_ms >= 45000 then return false end
    local output = now-arg.age_ms
    if t.last_output_ms == nil or t.last_output_ms == cjson.null or output > t.last_output_ms then
        t.last_output_ms = output
    end
elseif op == 'authorize' then
    if external then return false end
    if not owned or not t.publish_allowed or t.desired_state ~= 'run' then return false end
    t.sessions[arg.session_id] = t.generation
elseif op == 'freeze' then
    if external then return false end
    if not owned and t.desired_state ~= 'stopped' then return false end
    t.publish_allowed, t.ready = false, false
elseif op == 'clear' then
    if external then return false end
    if not owned and t.desired_state ~= 'stopped' then return false end
    for _,id in ipairs(arg.sessions) do t.sessions[id] = nil end
    if t.desired_state == 'stopped' and t.lease_until <= now and next(t.sessions) == nil then
        t.state, t.owner_id, t.lease_until = 'stopped', cjson.null, 0
        redis.call('SREM',pool,t.id)
    end
elseif op == 'external_close' then
    if not external or terminal then return false end
    if arg.process_exited then
        if arg.owner_id == cjson.null or t.publisher_owner_id ~= arg.owner_id or t.generation ~= arg.generation then
            return false
        end
    else
        if t.publisher_owner_id ~= cjson.null then return false end
        if not owned and t.desired_state ~= 'stopped' then return false end
    end
    t.publisher_owner_id, t.publish_allowed, t.ready = cjson.null, false, false
    if t.desired_state == 'stopped' then
        t.state, t.owner_id, t.lease_until = 'stopped', cjson.null, 0
        redis.call('SREM', pool, t.id)
    end
elseif op == 'open' then
    if not owned or t.desired_state ~= 'run' or next(t.sessions) ~= nil then return false end
    if external then
        if t.publisher_owner_id ~= cjson.null then return false end
        t.publisher_owner_id = arg.owner_id
    end
    t.publish_allowed = true
else
    return redis.error_reply('Unknown task operation')
end
t.sequence = t.sequence or 1
local changed = t.state ~= old_state or t.ready ~= old_ready or t.desired_state ~= old_desired
if changed then t.sequence = t.sequence + 1 end
t.updated_at = ARGV[5]
t._observed_ms = now
local result = cjson.encode(t)
redis.call('HSET', KEYS[1], 'payload', result)
local final = t.state == 'succeeded' or t.state == 'failed' or t.state == 'cancelled' or t.state == 'stopped'
if changed and t.notification_target and t.notification_target ~= cjson.null and (t.kind == 'stream' or final) then
    local key = 'vs:notification:'..t.id
    local lease = tonumber(redis.call('HGET',key,'lease_until') or '0')
    redis.call('HSET',key,'task',result,'sequence',t.sequence,'attempts',0)
    redis.call('ZADD','vs:notifications:due:'..t.notification_target,math.max(now,lease),t.id)
end
return result
"""


NOTIFICATION = """
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
local op, arg = ARGV[1], cjson.decode(ARGV[2])
local due = 'vs:notifications:due:'..arg.target
local inflight = 'vs:notifications:inflight:'..arg.target
local global = 'vs:notifications:inflight'
redis.call('ZREMRANGEBYSCORE',inflight,'-inf',now)
redis.call('ZREMRANGEBYSCORE',global,'-inf',now)
if op == 'claim' then
    local count = math.min(arg.limit,2-redis.call('ZCARD',inflight),4-redis.call('ZCARD',global))
    if count <= 0 then return '[]' end
    local ids = redis.call('ZRANGEBYSCORE',due,'-inf',now,'LIMIT',0,count)
    local events = {}
    for _,id in ipairs(ids) do
        local key = 'vs:notification:'..id
        local raw = redis.call('HGET',key,'task')
        if raw then
            local token = arg.token..':'..id
            local lease = now+30000
            redis.call('HSET',key,'token',token,'lease_until',lease)
            redis.call('ZADD',due,lease,id)
            redis.call('ZADD',inflight,lease,id)
            redis.call('ZADD',global,lease,id)
            table.insert(events,{task=cjson.decode(raw),token=token,
                attempts=tonumber(redis.call('HGET',key,'attempts') or '0')})
        else redis.call('ZREM',due,id) end
    end
    if #events == 0 then return '[]' end
    return cjson.encode(events)
end
local key = 'vs:notification:'..arg.id
if redis.call('HGET',key,'token') ~= arg.token then return 0 end
redis.call('ZREM',inflight,arg.id)
redis.call('ZREM',global,arg.id)
redis.call('HSET',key,'lease_until',0,'token','')
if tonumber(redis.call('HGET',key,'sequence')) ~= arg.sequence then
    redis.call('ZADD',due,now,arg.id)
    return 0
end
if arg.delivered then
    local t = cjson.decode(redis.call('HGET',key,'task'))
    redis.call('ZREM',due,arg.id)
    redis.call('DEL',key)
    if t.state == 'succeeded' or t.state == 'failed' or t.state == 'cancelled' or t.state == 'stopped' then
        redis.call('SREM','vs:notifications:outstanding',arg.id)
    end
else
    redis.call('HINCRBY',key,'attempts',1)
    redis.call('ZADD',due,now+arg.delay_ms,arg.id)
end
return 1
"""


def request_fingerprint(**values) -> str:
    """按有效参数和输入内容生成稳定指纹，不包含本地随机路径。"""
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def publisher_exit_pending(record: dict) -> bool:
    """外部发布者未确认退出时，任务不能进入可接管队列。"""
    return bool(record.get("publish_url") and record["generation"] > 0 and
                ("publisher_owner_id" not in record or record["publisher_owner_id"] is not None))


def stream_health(record: dict, now_ms: int) -> dict | None:
    """用当前 owner 租约和当前代次实际输出证明播放有效期。"""
    if record["kind"] != "stream":
        return None
    output = record.get("last_output_ms")
    lease_valid = bool(record.get("owner_id") and record["lease_until"] > now_ms)
    valid_until = min(record["lease_until"], output + 45000) if lease_valid and output else None
    def timestamp(value):
        return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat() if value else None
    return dict(observed_at=timestamp(now_ms), last_output_at=timestamp(output),
                valid_until=timestamp(valid_until), lease_valid=lease_valid, generation=record["generation"])


class CapacityExceeded(Exception):
    pass


class InvalidTransition(Exception):
    pass


class IdempotencyConflict(Exception):
    """同一意图键禁止被不同输入复用。"""


class TaskStore:
    """Atomic JSON records; ownership operations return None when their CAS loses."""
    def __init__(self, client: Redis, file_limit=100, stream_limit=100, lease_seconds=20, *, notification_limit=1000):
        self.client = client
        self.file_limit, self.stream_limit = file_limit, stream_limit
        self.lease_ms = int(lease_seconds * 1000)
        self.notification_limit = notification_limit

    async def _change(self, task_id, operation, arguments, limit=0):
        result = await self.client.eval(
            TRANSITION, 5, f"vs:task:{task_id}", "vs:active:files", "vs:active:streams",
            self._binding_key(arguments.get("submission_key") or "", arguments.get("notification_target")), "vs:notifications:outstanding",
            operation, json.dumps(arguments), self.lease_ms, limit,
            datetime.now(timezone.utc).isoformat(),
            self.notification_limit,
        )
        if result in (b"capacity", "capacity"):
            raise CapacityExceeded
        if result in (b"invalid", "invalid"):
            raise InvalidTransition
        if result in (b"conflict", "conflict"):
            raise IdempotencyConflict
        return json.loads(result) if result else None

    async def create(self, kind: str, source_kind: str, source_ref: str, task_id: str | None = None,
                     *, redaction_mode: str = "solid", profile_detection: bool | str = False, detection_interval: int = 1,
                     tracking: str = "template",
                     idempotency_key: str | None = None, fingerprint: str | None = None,
                     notification_target: str | None = None, publish_url: str | None = None,
                     playback_url: str | None = None,
                     max_bitrate: int | None = None, audio_enabled: bool = True) -> dict:
        if redaction_mode not in {"solid", "mosaic"}:
            raise ValueError("Unsupported redaction mode")
        if profile_detection and kind != 'video':
            raise ValueError('Detection profiling requires a video task')
        if profile_detection not in (False, True, 'host', 'forward-primary-face', 'forward-primary-plate',
                                     'forward-detail-face', 'forward-detail-plate', 'local-face'):
            raise ValueError('Unsupported detection profile')
        if (type(detection_interval) is not int or detection_interval not in (1, 3)
                or detection_interval != 1 and kind not in ('video', 'stream')):
            raise ValueError('Unsupported detection interval')
        if tracking not in ('template', 'flow') or tracking == 'flow' and (kind not in ('video', 'stream') or detection_interval != 3):
            raise ValueError('Unsupported tracking mode')
        if publish_url is not None and kind != "stream":
            raise ValueError("A publish URL requires a stream task")
        if playback_url is not None and kind != 'stream':
            raise ValueError('A playback URL requires a stream task')
        if max_bitrate is not None and (kind != "stream" or type(max_bitrate) is not int or max_bitrate < 1):
            raise ValueError("A max bitrate requires a positive integer stream task")
        if type(audio_enabled) is not bool or not audio_enabled and kind != "stream":
            raise ValueError("Invalid stream audio switch")
        task_id = task_id or str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        task = dict(
            id=task_id, kind=kind, redaction_mode=redaction_mode, state="queued", desired_state="run", ready=False,
            created_at=now, updated_at=now, source_kind=source_kind, source_ref=source_ref,
            result_ref=None, owner_id=None, generation=0, lease_until=0,
            dispatch_pending=True, dispatch_until=0, error=None, publish_allowed=False,
            sessions={}, stats=dict(processed_frames=0, dropped_frames=0, elapsed_seconds=0),
            sequence=1, last_output_ms=None, submission_key=idempotency_key,
            fingerprint=fingerprint, notification_target=notification_target,
        )
        if profile_detection:
            task['profile_detection'] = profile_detection
        if kind == 'stream' or detection_interval != 1:
            task['detection_interval'] = detection_interval
        if kind == 'stream' or tracking != 'template':
            task['tracking'] = tracking
        if publish_url is not None:
            task["publish_url"] = publish_url
            task["publisher_owner_id"] = None
        if playback_url is not None:
            task['playback_url'] = playback_url
        if max_bitrate is not None:
            task["max_bitrate"] = max_bitrate
        if not audio_enabled:
            task["audio_enabled"] = False
        return await self._change(task_id, "create", task, self.stream_limit if kind == "stream" else self.file_limit)

    async def get(self, task_id: str) -> dict | None:
        async with self.client.pipeline(transaction=True) as commands:
            commands.hget(f"vs:task:{task_id}", "payload")
            commands.time()
            payload, clock = await commands.execute()
        return json.loads(payload) | {"_observed_ms": clock[0] * 1000 + clock[1] // 1000} if payload else None

    @staticmethod
    def _binding_key(key: str, notification_target: str | None = None) -> str:
        """键哈希避免外部文本成为 Redis 路径。"""
        prefix = "vs:idempotency:"
        # 旧目标沿用原绑定；通用目标分别保存业务幂等键，避免不同环境互相冲突。
        if notification_target not in (None, "platform", "event-hub"):
            prefix += "target:"
            key = json.dumps([notification_target, key], separators=(",", ":"))
        return prefix + hashlib.sha256(key.encode()).hexdigest()

    async def replay(self, key: str, fingerprint: str, *, notification_target: str | None = None) -> dict | None:
        """在 worker 可用性与新容量检查前找回已接纳任务。"""
        task_id = await self.client.get(self._binding_key(key, notification_target))
        if not task_id:
            return None
        record = await self.get(task_id.decode() if isinstance(task_id, bytes) else task_id)
        if not record or record.get("fingerprint") != fingerprint:
            raise IdempotencyConflict
        return record

    async def query(self, task_ids: list[str]) -> list[dict]:
        """单次 pipeline 有界读取，终态不依赖活跃集合。"""
        if not 1 <= len(task_ids) <= 100:
            raise ValueError("任务查询数量必须在 1 到 100 之间")
        async with self.client.pipeline(transaction=False) as commands:
            for task_id in dict.fromkeys(task_ids):
                commands.hget(f"vs:task:{task_id}", "payload")
            values = await commands.execute()
        return [json.loads(raw) for raw in values if raw]

    async def health(self, record: dict) -> dict | None:
        """只读 Redis 时间，不延长 owner 租约或输出有效期。"""
        if record["kind"] != "stream":
            return None
        seconds, micros = await self.client.time()
        return stream_health(record, seconds * 1000 + micros // 1000)

    async def mark_output(self, task_id: str, owner_id: str, generation: int, *, age_ms=0):
        """只接受当前发布者实际写出画面后的合并心跳。"""
        return await self._change(task_id, "output", dict(owner_id=owner_id, generation=generation, age_ms=age_ms))

    async def publisher_change(self, task_id: str, operation: str, **arguments):
        """媒体侧变更复用统一 Lua，避免绕过公共序号与通知。"""
        return await self._change(task_id, operation, arguments)

    async def claim_notifications(self, *, limit=4, targets=("event-hub", "platform")) -> list[dict]:
        """全局四项、每目标两项，租约到期后其他 notifier 可接管。"""
        events = []
        for target in targets:
            if len(events) >= limit:
                break
            raw = await self.client.eval(NOTIFICATION, 0, "claim", json.dumps(
                dict(target=target, limit=min(4, limit - len(events)), token=str(uuid4()))))
            events.extend(json.loads(raw))
        return events

    async def complete_notification(self, event: dict, *, delivered: bool, delay_seconds=1) -> bool:
        """旧 token/序号不能清除新状态；确认仅交接责任，不删除成品。"""
        task = event["task"]
        result = await self.client.eval(NOTIFICATION, 0, "complete", json.dumps(dict(
            target=task["notification_target"], id=task["id"], token=event["token"],
            sequence=task["sequence"], delivered=delivered, delay_ms=int(delay_seconds * 1000))))
        return bool(result)

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
