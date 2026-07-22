"""Optional Redis helper with graceful degradation when unavailable."""

from __future__ import annotations

from typing import Any

from smartops.core.logging import get_logger

logger = get_logger(__name__)


def create_redis_client(redis_url: str | None) -> Any | None:
    """Return a sync Redis client, or None when URL empty / connection fails."""
    url = (redis_url or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore
    except ImportError:
        logger.warning("redis_package_missing")
        return None
    try:
        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.5)
        client.ping()
        logger.info("redis_connected", url=url.split("@")[-1])
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis_unavailable", error=str(exc))
        return None
