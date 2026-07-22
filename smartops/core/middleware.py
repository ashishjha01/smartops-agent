"""HTTP middleware: request IDs, timing, Redis/in-memory rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from smartops.core.async_utils import run_sync
from smartops.core.logging import get_logger

logger = get_logger(__name__)

_RATE_LIMIT_EXEMPT = {"/health", "/ready", "/metrics", "/", "/docs", "/openapi.json", "/redoc"}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Adds request IDs and timing headers using the app's configured header name."""

    def __init__(self, app, request_id_header: str = "X-Request-ID"):
        super().__init__(app)
        self.request_id_header = request_id_header

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(self.request_id_header) or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers[self.request_id_header] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_s=round(elapsed, 4),
            request_id=request_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP.

    Uses Redis when a client is provided (multi-instance safe for rate limits);
    otherwise falls back to process-local memory.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        redis_client: Any | None = None,
        redis_key_prefix: str = "smartops:ratelimit",
        trust_proxy_headers: bool = False,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.redis = redis_client
        self.redis_key_prefix = redis_key_prefix
        self.trust_proxy_headers = trust_proxy_headers
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self.backend = "redis" if redis_client is not None else "memory"

    def _client_key(self, request: Request) -> str:
        # Never trust client-supplied XFF unless explicitly behind a known proxy.
        if self.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip() or "unknown"
        return request.client.host if request.client else "unknown"

    def _allow_memory(self, client: str) -> bool:
        now = time.time()
        if len(self._hits) > 512:
            stale = [k for k, w in list(self._hits.items()) if (not w) or (now - w[-1] > 60)]
            for key in stale[:128]:
                self._hits.pop(key, None)
        window = self._hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.requests_per_minute:
            return False
        window.append(now)
        return True

    def _allow_redis(self, client: str) -> bool:
        assert self.redis is not None
        now = time.time()
        key = f"{self.redis_key_prefix}:{client}"
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - 60)
        pipe.zadd(key, {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = pipe.execute()
        count = int(results[2])
        return count <= self.requests_per_minute

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _RATE_LIMIT_EXEMPT or request.url.path.startswith("/docs"):
            return await call_next(request)

        client = self._client_key(request)
        try:
            if self.redis is not None:
                allowed = await run_sync(self._allow_redis, client)
            else:
                allowed = self._allow_memory(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rate_limit_redis_failed", error=str(exc))
            allowed = self._allow_memory(client)

        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": "60"},
                content={
                    "detail": "Rate limit exceeded. Try again shortly.",
                    "retry_after_seconds": 60,
                    "backend": self.backend,
                },
            )
        return await call_next(request)
