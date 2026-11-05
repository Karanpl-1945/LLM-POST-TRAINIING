from __future__ import annotations

import os
import time

try:
    import redis.asyncio as redis
except ImportError:  # optional in unit tests
    redis = None


class RedisFixedWindowLimiter:
    """Application-level backup limiter. Put Nginx/API-gateway limiting in front as well."""

    def __init__(self, requests_per_minute: int = 60):
        self.limit = requests_per_minute
        self.url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.from_url(self.url, decode_responses=True) if redis else None

    async def allow(self, key: str) -> bool:
        if self.client is None:
            return True
        window = int(time.time() // 60)
        redis_key = f"rate:{key}:{window}"
        count = await self.client.incr(redis_key)
        if count == 1:
            await self.client.expire(redis_key, 70)
        return count <= self.limit
