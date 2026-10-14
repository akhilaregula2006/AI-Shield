"""In-memory sliding-window rate limiter (per client IP). Swap for Redis if scaled horizontally."""

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status

from configs.settings import settings


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit, self.window = limit, window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> Tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(int(self.window - (now - hits[0])) + 1, 1)
            hits.append(now)
            return True, 0


_limiter: Optional[SlidingWindowLimiter] = None


def get_limiter() -> SlidingWindowLimiter:
    global _limiter  # pylint: disable=global-statement
    if _limiter is None:
        _limiter = SlidingWindowLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
    return _limiter


def reset_limiter() -> None:
    global _limiter  # pylint: disable=global-statement
    _limiter = None


def rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 when the caller exceeds the configured request rate."""
    key = request.client.host if request.client else "unknown"
    allowed, retry_after = get_limiter().allow(key)
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
