"""Taskiq Redis streams with bounded admission and explicit durable ACK."""

from redis.asyncio import Redis
from taskiq import AckableMessage
from taskiq_redis import RedisStreamBroker

from vision_shield.config import Settings
from vision_shield.jobs.store import CapacityExceeded

ENQUEUE = """
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[1]) then return false end
local t, key
if ARGV[3] and ARGV[3] ~= '' then
    key = 'vs:task:' .. ARGV[3]
    local raw = redis.call('HGET', key, 'payload')
    if not raw then return 'skipped' end
    t = cjson.decode(raw)
    local clock = redis.call('TIME')
    local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
    if t.desired_state ~= 'run' or (t.state ~= 'queued' and t.state ~= 'recovering') then return 'skipped' end
    if t.lease_until > now or t.dispatch_until > now then return 'skipped' end
    t.dispatch_until, t.dispatch_pending = now + 10000, false
end
local id = redis.call('XADD', KEYS[1], '*', 'data', ARGV[2])
if t then redis.call('HSET', key, 'payload', cjson.encode(t)) end
return id
"""
ACK_DELETE = """
redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
return redis.call('XDEL', KEYS[1], ARGV[2])
"""


class BoundedRedisBroker(RedisStreamBroker):
    def __init__(self, *args, message_limit=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_limit = message_limit
        self.pending_acks = {}

    async def kick(self, message):
        async with Redis(connection_pool=self.connection_pool) as client:
            result = await client.eval(ENQUEUE, 1, self.queue_name, self.message_limit,
                                       message.message, message.labels.get("job_id", ""))
            if result is None:
                raise CapacityExceeded

    def _ack_generator(self, id, queue_name):
        async def ack():
            async with Redis(connection_pool=self.connection_pool) as client:
                await client.eval(ACK_DELETE, 1, queue_name, self.consumer_group_name, id)
        return ack

    async def listen(self):
        async with Redis(connection_pool=self.connection_pool) as client:
            while True:
                batches = await client.xreadgroup(self.consumer_group_name, self.consumer_name,
                                                 {self.queue_name: ">"}, count=1, block=1000)
                messages = [(stream, entry_id, fields) for stream, batch in batches for entry_id, fields in batch]
                if not messages:
                    # Also reclaim after an empty read; an idle queue must recover a lost delivery.
                    _, pending, *_ = await client.xautoclaim(self.queue_name, self.consumer_group_name,
                                                             self.consumer_name, self.idle_timeout, count=1)
                    messages = [(self.queue_name, entry_id, fields) for entry_id, fields in pending]
                for stream, entry_id, fields in messages:
                    payload = fields[b"data"]
                    task_id = self.formatter.loads(payload).task_id
                    callback = self._ack_generator(entry_id, stream)

                    async def ack(cb=callback, message_id=task_id):
                        await cb()
                        self.pending_acks.pop(message_id, None)

                    self.pending_acks[task_id] = ack
                    yield AckableMessage(data=payload, ack=ack)

    async def ack_message(self, task_id):
        await self.pending_acks[task_id]()


settings = Settings()
file_broker = BoundedRedisBroker(settings.redis_url.get_secret_value(), queue_name="vs:files",
                                consumer_group_name="vs:files", consumer_id="0",
                                message_limit=settings.message_limit, xread_count=1,
                                socket_connect_timeout=2, socket_timeout=2)
stream_broker = BoundedRedisBroker(settings.redis_url.get_secret_value(), queue_name="vs:streams",
                                  consumer_group_name="vs:streams", consumer_id="0",
                                  message_limit=settings.message_limit, xread_count=1,
                                  socket_connect_timeout=2, socket_timeout=2)
