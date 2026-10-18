from fastapi import HTTPException
from redis.asyncio import Redis


async def require_worker(client: Redis, kind: str, lease_seconds: float) -> None:
    pool = "streams" if kind == "stream" else "files"
    seconds, micros = await client.time()
    now = seconds * 1000 + micros // 1000
    async with client.pipeline(transaction=True) as commands:
        commands.zremrangebyscore(f"vs:workers:{pool}", "-inf", now - int(lease_seconds * 1000))
        commands.zcard(f"vs:workers:{pool}")
        _, available = await commands.execute()
    if not available:
        raise HTTPException(503, "dependency_unavailable")
